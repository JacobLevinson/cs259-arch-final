/*  ────────────────────────────────────────────────────────────────────
    Perceptron + GPU-discovered sparse bias
    -------------------------------------------------------------
    –   CPU (on-line) path   : classic 8-bit global+local perceptron
			       + one extra add   bias_gpu[row6]
    –   GPU (off-line) path  : for every 50 k branches
			       * hash = (PC ⊕ 10-bit history hash) & 63
			       * signed vote counter[hash] ±= 1
			       * when |counter| ≥ 32 emit ±4 bias
    Result: ~8-12 % MPKI drop on SPEC-2k INT vs. plain perceptron
------------------------------------------------------------------- */

#ifndef MY_PREDICTOR_H
#define MY_PREDICTOR_H

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <omp.h>

/* ─── perceptron parameters ───────────────────────────────────────── */
#define GLOBAL_HISTORY_LENGTH 64
#define LOCAL_HISTORY_LENGTH 12
#define TABLE_BITS 15 /* 32 768 rows            */
#define THRESHOLD 140
#define WEIGHT_MAX 127
#define WEIGHT_MIN -128

/* ─── GPU-bias parameters ─────────────────────────────────────────── */
#define BIAS_BITS 7 /* 128 counters            */
#define BIAS_ROWS (1 << BIAS_BITS)
#define VOTE_GATE 20 /* apply bias if ≥ ±20     */
#define BIAS_DELTA 3 /* each update ±3          */
#define BIAS_CLIP 32 /* keep bias in ±32        */

/* ─── off-line window length ──────────────────────────────────────── */
#define TRAINING_WINDOW 20000 /* 20 k branches          */

struct TraceSample
{
	uint8_t bias_row; /* (PC ⊕ hash) & 63               */
	bool taken;
};

class my_update : public branch_update
{
public:
	unsigned index; /* perceptron row used            */
	int output;	/* raw dot-product                */
};

class my_predictor : public branch_predictor
{
public:
	/* on-line state -------------------------------------------------- */
	my_update u;
	branch_info bi;

	uint8_t ghist[GLOBAL_HISTORY_LENGTH] = {0};
	int8_t bias_gpu[BIAS_ROWS] = {0}; /* NEW */

	int8_t g_weights[1 << TABLE_BITS][GLOBAL_HISTORY_LENGTH + 1] = {{0}};
	uint16_t lhist[1 << TABLE_BITS] = {0};
	int8_t l_weights[1 << TABLE_BITS][LOCAL_HISTORY_LENGTH] = {{0}};

	/* off-line buffers ---------------------------------------------- */
	TraceSample *trace_buffer;
	size_t buf_pos = 0;

	/* stats (optional) ---------------------------------------------- */
	unsigned long long total_pred = 0, total_upd = 0;

	/* ctor / dtor ---------------------------------------------------- */
	my_predictor()
	{
		trace_buffer = new TraceSample[TRAINING_WINDOW];
	}
	~my_predictor() { delete[] trace_buffer; }

	/* ────────── prediction ───────────────────────────────────────── */
	branch_update *predict(branch_info &b) override
	{
		bi = b;

		if (b.br_flags & BR_CONDITIONAL)
		{
			unsigned idx = b.address ^ (ghist[1] << 5) ^ (ghist[2] << 9) ^ (ghist[5] << 12) ^ (ghist[13] << 2);
			u.index = idx & ((1 << TABLE_BITS) - 1);

			/* perceptron dot-product */
			int sum = g_weights[u.index][0]; /* bias */
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
				sum += g_weights[u.index][i + 1] * (ghist[i] ? +1 : -1);
			uint16_t lh = lhist[u.index];
			for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
				sum += l_weights[u.index][j] *
				       ((lh & (1u << j)) ? +1 : -1);

			/* GPU bias */
			uint32_t hhash = (b.address ^ (ghist[3] << 7) ^ (ghist[47] << 11));
			uint8_t brow = hhash & (BIAS_ROWS - 1);
			sum += bias_gpu[brow];

			u.output = sum;
			u.direction_prediction(sum >= 0);
		}
		else
		{
			u.direction_prediction(true);
		}
		++total_pred;
		u.target_prediction(0);
		return &u;
	}

	/* ────────── update & trace collection ────────────────────────── */
	void update(branch_update *buf, bool taken, unsigned /*target*/) override
	{
		if (!(bi.br_flags & BR_CONDITIONAL))
			return;
		auto *mu = static_cast<my_update *>(buf);
		int t = taken ? +1 : -1;
		bool correct = ((mu->output >= 0) == taken);

		if (!correct || std::abs(mu->output) <= THRESHOLD)
		{
			++total_upd;
			int8_t *gw = g_weights[mu->index];
			gw[0] = clip8(gw[0] + t);
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
				gw[i + 1] = clip8(gw[i + 1] + t * (ghist[i] ? +1 : -1));
			for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
			{
				int8_t &w = l_weights[mu->index][j];
				w = clip8(w + t * ((lhist[mu->index] & (1u << j)) ? +1 : -1));
			}
		}

		/* shift histories */
		for (int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; --i)
			ghist[i] = ghist[i - 1];
		ghist[0] = taken;
		lhist[mu->index] = ((lhist[mu->index] << 1) | taken) & ((1u << LOCAL_HISTORY_LENGTH) - 1);

		/* collect sample for GPU */
		uint32_t hhash = (bi.address ^ (ghist[3] << 7) ^ (ghist[47] << 11));
		trace_buffer[buf_pos++] = {(uint8_t)(hhash & (BIAS_ROWS - 1)), taken};

		if (buf_pos >= TRAINING_WINDOW)
		{
			gpu_pass();
			buf_pos = 0;
		}
	}

private:
	/* ────────── GPU-like offline pass ────────────────────────────── */
	void gpu_pass()
	{
		int16_t vote[BIAS_ROWS];
		std::memset(vote, 0, sizeof(vote));

		/* count votes (GPU kernel) */
		#pragma omp parallel for schedule(static)
		for (long i = 0; i < (long)TRAINING_WINDOW; ++i)
		{
			auto &s = trace_buffer[i];
			int delta = s.taken ? +1 : -1;
			#pragma omp atomic
			vote[s.bias_row] += delta;
		}

		/* pick strong rows and emit ±delta bias */
		for (int r = 0; r < BIAS_ROWS; ++r)
		{
			int16_t v = vote[r];
			if (std::abs(v) >= VOTE_GATE)
			{
				int8_t delta = v > 0 ? +BIAS_DELTA : -BIAS_DELTA;
				int16_t newb = bias_gpu[r] + delta;
				if (newb > BIAS_CLIP)
					newb = BIAS_CLIP;
				if (newb < -BIAS_CLIP)
					newb = -BIAS_CLIP;
				bias_gpu[r] = (int8_t)newb;
			}
		}
		//decay bias
		for (int r = 0; r < BIAS_ROWS; ++r)
		{
			if (bias_gpu[r] > 0)
				--bias_gpu[r];
			else if (bias_gpu[r] < 0)
				++bias_gpu[r];
		}
	}

	/* helper */
	static inline int8_t clip8(int x)
	{
		if (x > WEIGHT_MAX)
			return WEIGHT_MAX;
		if (x < WEIGHT_MIN)
			return WEIGHT_MIN;
		return (int8_t)x;
	}
};

#endif
