// my_predictor.h
/*
  Perceptron branch predictor (global-plus-local) with a periodic “GPU”‐like
  offline retraining using a small MLP (77‐input → 16 hidden → 1 output).
  After each window of samples, the MLP is trained on those samples, then
  “distills” corrections into the 8‐bit perceptron tables. A simulated delay
  in branch counts models the time taken to do that training.
*/

#ifndef MY_PREDICTOR_H
#define MY_PREDICTOR_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// ─ Tunable parameters for the perceptron
#define GLOBAL_HISTORY_LENGTH 64 // number of bits in shared global history
#define LOCAL_HISTORY_LENGTH 12	 // bits in each branch’s local history
#define TABLE_BITS 15		 // 2^15 = 32,768 entries in weight table
#define THRESHOLD 140		 // margin under which we retrain
#define WEIGHT_MAX 127		 // 8-bit signed
#define WEIGHT_MIN -128

// Number of samples to collect before retraining
#define TRAINING_WINDOW 50000 // branches per window
// Simulated delay (in branches) while “GPU” trains the MLP
#define TRAINING_DELAY 50 //  branches of delay before applying nudges

// Dimensions for student (perceptron) vs teacher (MLP)
static constexpr int STUDENT_DIMS = 1 + GLOBAL_HISTORY_LENGTH + LOCAL_HISTORY_LENGTH; // 1 bias + 64 global + 12 local = 77
static constexpr int TEACHER_IN = GLOBAL_HISTORY_LENGTH + LOCAL_HISTORY_LENGTH;	      // 64 + 12 = 76
static constexpr int TEACHER_HID = 32;						      // hidden‐layer size for the MLP (smaller to keep training cost down)

// A single branch‐trace sample for offline retraining
struct TraceSample
{
	uint16_t index; // row index [0 .. 32767]
	uint64_t ghist; // packed 64 global bits (bit i = ghist’s i-th bit)
	uint16_t lhist; // packed 12 local bits (bit i = local history bit i)
	bool taken;	// actual outcome
};

class my_update : public branch_update
{
public:
	unsigned int index; // which row we used
	int output;	    // raw perceptron sum
};

class my_predictor : public branch_predictor
{
public:
	// ─ Public state (per‐branch, “hardware” path)
	my_update u;
	branch_info bi;

	int ghist[GLOBAL_HISTORY_LENGTH];
	// global history bits (0 or 1)

	int g_weights[1 << TABLE_BITS][GLOBAL_HISTORY_LENGTH + 1];
	// g_weights[row][0] = bias
	// g_weights[row][1..64] = weights for ghist[0..63], signed 8-bit in practice

	unsigned char lhist[1 << TABLE_BITS];
	// 12-bit local shift‐register per row, stored in low 12 bits

	int l_weights[1 << TABLE_BITS][LOCAL_HISTORY_LENGTH];
	// l_weights[row][0..11] = local weights, signed 8-bit

	// ─ Statistics
	unsigned long long total_predictions = 0;
	unsigned long long total_updates = 0;
	unsigned long long weak_predictions = 0;
	unsigned long long strong_correct = 0;
	unsigned long long strong_wrong = 0;

	// ─ Offline retraining members
	TraceSample *trace_buffer; // ring buffer of size TRAINING_WINDOW
	size_t buffer_pos;	   // next index to write in trace_buffer

	bool nudges_pending;	  // true if teacher‐driven deltas are ready to apply
	int training_delay_count; // simulated “branches to wait” before applying deltas

	// Student‐side delta table (per row, 77 signed 8-bit corrections from teacher)
	int8_t *row_deltas; // length = NUM_ROWS * STUDENT_DIMS

	// Accumulators (32-bit) for per‐row, per‐weight gradient sums
	int32_t *accumulators; // length = NUM_ROWS * STUDENT_DIMS

	// Teacher (MLP) parameters
	float teacher_w1[TEACHER_HID][TEACHER_IN];
	float teacher_b1[TEACHER_HID];
	float teacher_w2[TEACHER_HID];
	float teacher_b2;

	// Constructor / Destructor
	my_predictor()
	{
		// Initialize perceptron tables & histories to zero
		memset(ghist, 0, sizeof(ghist));
		memset(g_weights, 0, sizeof(g_weights));
		memset(lhist, 0, sizeof(lhist));
		memset(l_weights, 0, sizeof(l_weights));

		// Allocate and zero the trace buffer
		trace_buffer = new TraceSample[TRAINING_WINDOW];
		buffer_pos = 0;

		// Initialize teacher weight arrays with small random values
		std::mt19937 rng(12345); // fixed seed for reproducibility
		std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

		for (int h = 0; h < TEACHER_HID; ++h)
		{
			for (int i = 0; i < TEACHER_IN; ++i)
			{
				teacher_w1[h][i] = dist(rng);
			}
			teacher_b1[h] = dist(rng);
			teacher_w2[h] = dist(rng);
		}
		teacher_b2 = dist(rng);

		// Allocate and zero accumulators and deltas
		size_t num_rows = (1 << TABLE_BITS);
		size_t stu_size = (size_t)num_rows * STUDENT_DIMS;

		accumulators = (int32_t *)std::malloc(sizeof(int32_t) * stu_size);
		row_deltas = (int8_t *)std::malloc(sizeof(int8_t) * stu_size);

		if (!accumulators || !row_deltas)
		{
			std::fprintf(stderr, "ERROR: Failed to allocate offline buffers\n");
			std::exit(1);
		}
		std::memset(accumulators, 0, sizeof(int32_t) * stu_size);
		std::memset(row_deltas, 0, sizeof(int8_t) * stu_size);

		nudges_pending = false;
		training_delay_count = 0;
	}

	~my_predictor()
	{
		// Dump stats
		FILE *f = std::fopen("perceptron_stats.txt", "a");
		if (f)
		{
			std::fprintf(f,
				     "\n=== Perceptron Predictor Statistics ===\n"
				     "Total Predictions: %llu\n"
				     "Total Updates (training events): %llu\n"
				     "Weak Predictions (|output| <= threshold): %llu\n"
				     "Strong Correct Predictions: %llu\n"
				     "Strong Wrong Predictions: %llu\n"
				     "========================================\n",
				     total_predictions, total_updates, weak_predictions,
				     strong_correct, strong_wrong);
			std::fclose(f);
		}

		// Free buffers
		delete[] trace_buffer;
		std::free(accumulators);
		std::free(row_deltas);
	}

	// ─────────────────────────────────────────────────────────────────────
	// Called on every branch to get a prediction
	branch_update *predict(branch_info &b)
	{
		// If a retraining is “in flight,” decrement its delay counter
		if (training_delay_count > 0)
		{
			--training_delay_count;
			if (training_delay_count == 0 && nudges_pending)
			{
				apply_nudges();
				nudges_pending = false;
			}
		}

		bi = b;

		if (b.br_flags & BR_CONDITIONAL)
		{
			// Hash in a few GHIST bits to reduce collisions
			unsigned idx = b.address ^ (ghist[1] << 5) ^ (ghist[2] << 9) ^ (ghist[5] << 12) ^ (ghist[13] << 2);
			u.index = idx & ((1 << TABLE_BITS) - 1);

			// Compute perceptron sum: bias + Σ(global_i * ghist[i]) + Σ(local_j * lhist_bit[j])
			int sum = g_weights[u.index][0]; // bias
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
			{
				int bit = (ghist[i] ? +1 : -1);
				sum += g_weights[u.index][i + 1] * bit;
			}
			unsigned char local_hist = lhist[u.index];
			for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
			{
				int bit = ((local_hist & (1u << j)) ? +1 : -1);
				sum += l_weights[u.index][j] * bit;
			}

			u.output = sum;
			u.direction_prediction(sum >= 0);

			++total_predictions;
			if (std::abs(sum) <= THRESHOLD)
			{
				++weak_predictions;
			}
		}
		else
		{
			// Unconditional or other branch → predict “taken”
			u.direction_prediction(true);
		}

		u.target_prediction(0);
		return &u;
	}

	// ─────────────────────────────────────────────────────────────────────
	// Called when actual outcome is known; does normal perceptron training
	// plus buffering for offline retraining
	void update(branch_update *buf, bool taken, unsigned /*target*/)
	{
		if (!(bi.br_flags & BR_CONDITIONAL))
		{
			return;
		}

		auto *mu = static_cast<my_update *>(buf);
		bool correct = ((mu->output >= 0) == taken);
		bool strong = (std::abs(mu->output) > THRESHOLD);
		if (strong)
		{
			(correct ? ++strong_correct : ++strong_wrong);
		}

		int t = taken ? +1 : -1;

		// Standard perceptron update if mispredicted or weak
		if (!correct || !strong)
		{
			++total_updates;
			int *gw = g_weights[mu->index];
			unsigned char local_hist = lhist[mu->index];
			int *lw = l_weights[mu->index];

			// Bias
			gw[0] = saturate(gw[0] + t);
			// Global weights
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
			{
				int bit = (ghist[i] ? +1 : -1);
				gw[i + 1] = saturate(gw[i + 1] + t * bit);
			}
			// Local weights
			for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
			{
				int bit = ((local_hist & (1u << j)) ? +1 : -1);
				lw[j] = saturate(lw[j] + t * bit);
			}
		}

		// Shift in the new global outcome
		for (int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; --i)
		{
			ghist[i] = ghist[i - 1];
		}
		ghist[0] = (taken ? 1 : 0);

		// Update local history for this row
		lhist[mu->index] = (unsigned char)(((lhist[mu->index] << 1) | (taken ? 1 : 0)) & ((1u << LOCAL_HISTORY_LENGTH) - 1));

		// ─────────────────────────────────────────────────────────────────
		// Buffer this sample for offline “GPU” retraining
		TraceSample &s = trace_buffer[buffer_pos++];
		s.index = (uint16_t)mu->index;

		// Pack global history bits
		uint64_t packed_gh = 0;
		for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
		{
			if (ghist[i])
			{
				packed_gh |= (uint64_t(1) << i);
			}
		}
		s.ghist = packed_gh;

		// Pack local history bits (low 12 bits)
		s.lhist = (uint16_t)(lhist[mu->index] & ((1u << LOCAL_HISTORY_LENGTH) - 1));
		s.taken = taken;

		// If buffer is full, start offline retraining
		if (buffer_pos >= TRAINING_WINDOW)
		{
			start_offline_training();
			buffer_pos = 0;
		}
	}

private:
	// Saturate to 8-bit signed
	static int saturate(int x)
	{
		if (x > WEIGHT_MAX)
			return WEIGHT_MAX;
		if (x < WEIGHT_MIN)
			return WEIGHT_MIN;
		return x;
	}

	// ─────────────────────────────────────────────────────────────────────
	// Called when TRAINING_WINDOW samples have been collected. We
	// train the teacher MLP on those samples, then compute per-row,
	// per-weight “deltas” (±1 or 0) to nudge the student. We record
	// them in row_deltas[], then schedule them to apply after a delay.
	void start_offline_training()
	{
		// 1) Train the teacher MLP on trace_buffer[0..TRAINING_WINDOW-1]
		train_teacher();

		// 2) Distill the teacher’s predictions into student deltas
		distill_teacher_to_student();

		// 3) Simulate a “GPU delay” before applying those deltas
		nudges_pending = true;
		training_delay_count = TRAINING_DELAY;
	}

	// ─────────────────────────────────────────────────────────────────────
	// Perform one or more epochs of SGD on the teacher MLP, using mean-squared
	// loss with tanh activations. We train on all TRAINING_WINDOW samples.
	void train_teacher()
	{
		const float lr = 0.01f;
		const int epochs = 3;

/* --- one OpenMP team for the whole routine ---------------------- */
#pragma omp parallel
		{
			/* thread-private scratch buffers live on the stack */
			float x_in[TEACHER_IN];
			float hidden[TEACHER_HID], dhidden[TEACHER_HID];

			/* thread-private gradient accumulators */
			alignas(64) float dw1[TEACHER_HID][TEACHER_IN] = {{0}};
			float db1[TEACHER_HID] = {0};
			float dw2[TEACHER_HID] = {0};
			float db2 = 0;

			for (int ep = 0; ep < epochs; ++ep)
			{
				/* zero thread-local grads each epoch */
				std::memset(dw1, 0, sizeof(dw1));
				std::memset(db1, 0, sizeof(db1));
				std::memset(dw2, 0, sizeof(dw2));
				db2 = 0;

/* ------------ parallel over the whole window ------------ */
#pragma omp for schedule(static)
				for (long idx = 0; idx < TRAINING_WINDOW; ++idx)
				{
					const TraceSample &s = trace_buffer[idx];

					/* ---- build input vector {±1} ------------------------ */
					unpack_teacher_input(s, x_in);

					const float y = s.taken ? 1.0f : -1.0f;

					/* ---- forward pass (vectorised dot products help) ---- */
					for (int h = 0; h < TEACHER_HID; ++h)
					{
						float z = teacher_b1[h];
						for (int i = 0; i < TEACHER_IN; ++i)
							z += teacher_w1[h][i] * x_in[i];
						hidden[h] = std::tanh(z);
					}
					float z_out = teacher_b2;
					for (int h = 0; h < TEACHER_HID; ++h)
						z_out += teacher_w2[h] * hidden[h];
					const float y_pred = std::tanh(z_out);

					/* ---- backward pass, **add to local grads** ---------- */
					const float d_out = (y_pred - y) * (1 - y_pred * y_pred);

					for (int h = 0; h < TEACHER_HID; ++h)
					{
						dw2[h] += d_out * hidden[h]; // ∂L/∂w2
					}
					db2 += d_out; // ∂L/∂b2

					for (int h = 0; h < TEACHER_HID; ++h)
					{
						const float g = d_out * teacher_w2[h] * (1 - hidden[h] * hidden[h]);
						db1[h] += g; // ∂L/∂b1
						for (int i = 0; i < TEACHER_IN; ++i)
							dw1[h][i] += g * x_in[i]; // ∂L/∂w1
					}
				} /* end for TRAINING_WINDOW */

/* ---- one thread at a time updates the real weights ---- */
#pragma omp critical
				{
					for (int h = 0; h < TEACHER_HID; ++h)
					{
						teacher_w2[h] -= lr * dw2[h];
						teacher_b1[h] -= lr * db1[h];
						for (int i = 0; i < TEACHER_IN; ++i)
							teacher_w1[h][i] -= lr * dw1[h][i];
					}
					teacher_b2 -= lr * db2;
				}

/* all threads wait before next epoch so weights are in sync */
#pragma omp barrier
			} /* end epochs */
		} /* end parallel region */
	}

	// ─────────────────────────────────────────────────────────────────────
	// After training the teacher, compare its outputs on each sample to the
	// student’s outputs. Accumulate ±1 “gradients” for any weight where the
	// teacher disagrees or the student’s margin is small. Then quantize each
	// row’s accumulator to {+1, 0, -1} per weight, storing into row_deltas[].
	void distill_teacher_to_student()
	{
		const size_t num_rows = (1u << TABLE_BITS);
		const size_t stu_size = num_rows * STUDENT_DIMS;
		std::memset(accumulators, 0, sizeof(int32_t) * stu_size);
/* -------- first pass: accumulate into thread-local buffers ---- */
#pragma omp parallel
		{
			int x_stu[STUDENT_DIMS];
			float x_tchr[TEACHER_IN], hidden[TEACHER_HID];

			/* each thread gets its own slice */
			std::vector<int32_t> acc_local(stu_size, 0);

#pragma omp for schedule(static)
			for (long idx = 0; idx < TRAINING_WINDOW; ++idx)
			{
				const TraceSample &s = trace_buffer[idx];
				const uint16_t row = s.index;

				unpack_teacher_input(s, x_tchr);

				/* teacher forward */
				for (int h = 0; h < TEACHER_HID; ++h)
				{
					float z = teacher_b1[h];
					for (int i = 0; i < TEACHER_IN; ++i)
						z += teacher_w1[h][i] * x_tchr[i];
					hidden[h] = std::tanh(z);
				}
				float z_out = teacher_b2;
				for (int h = 0; h < TEACHER_HID; ++h)
					z_out += teacher_w2[h] * hidden[h];
				const int t_sign = (z_out >= 0) ? 1 : -1;

				/* student forward (fast integer dot) */
				int sum_s = g_weights[row][0];
				for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
					sum_s += g_weights[row][i + 1] *
						 ((s.ghist & (1ull << i)) ? +1 : -1);
				for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
					sum_s += l_weights[row][j] *
						 ((s.lhist & (1u << j)) ? +1 : -1);

				const int margin = std::abs(sum_s);
				const int s_sign = (sum_s >= 0) ? 1 : -1;

				if (t_sign != s_sign || margin <= (THRESHOLD / 2))
				{
					build_student_input(s, x_stu);
					int32_t *acc_row = acc_local.data() + (size_t)row * STUDENT_DIMS;
					for (int k = 0; k < STUDENT_DIMS; ++k)
						acc_row[k] += t_sign * x_stu[k];
				}
			} /* end window loop */

/* ------ merge local→global once, outside the hot loop ------ */
#pragma omp critical
			for (size_t i = 0; i < stu_size; ++i)
				accumulators[i] += acc_local[i];
		} /* end parallel */

/* -------- second pass: quantise -------- */
#pragma omp parallel for schedule(static)
		for (long row = 0; row < (1 << TABLE_BITS); ++row)
		{
			int32_t *acc_row = accumulators + row * STUDENT_DIMS;
			int8_t *del_row = row_deltas + row * STUDENT_DIMS;
			for (int k = 0; k < STUDENT_DIMS; ++k)
				del_row[k] = (acc_row[k] > 0) - (acc_row[k] < 0); // {-1,0,+1}
		}
	}

	// ─────────────────────────────────────────────────────────────────────
	// Once the simulated delay expires, apply row_deltas[row][k] to student weights:
	// g_weights[row][0] += delta[0], g_weights[row][i+1] += delta[i+1], l_weights[row][j] += delta[65+j]
	void apply_nudges()
	{
		size_t num_rows = (1 << TABLE_BITS);
		for (size_t row = 0; row < num_rows; ++row)
		{
			int8_t *del_row = row_deltas + row * STUDENT_DIMS;

			// Bias weight
			if (del_row[0] != 0)
			{
				g_weights[row][0] = saturate(g_weights[row][0] + del_row[0]);
			}
			// Global weights [1..64]
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
			{
				int8_t d = del_row[i + 1];
				if (d != 0)
				{
					g_weights[row][i + 1] = saturate(g_weights[row][i + 1] + d);
				}
			}
			// Local weights [65..76]
			for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
			{
				int8_t d = del_row[1 + GLOBAL_HISTORY_LENGTH + j];
				if (d != 0)
				{
					l_weights[row][j] = saturate(l_weights[row][j] + d);
				}
			}
		}

		// After applying, we could zero out row_deltas if desired (not strictly needed)
		// std::memset(row_deltas, 0, sizeof(int8_t) * ((1<<TABLE_BITS) * STUDENT_DIMS));
	}

	// ─────────────────────────────────────────────────────────────────────
	// Unpack a TraceSample into the teacher’s input vector x_in[0..75]:
	// first 64 entries = +1/−1 from ghist bits, next 12 entries = +1/−1 from lhist bits.
	inline void unpack_teacher_input(const TraceSample &s, float *x_in)
	{
		// GHIST part
		for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
		{
			x_in[i] = ((s.ghist & (uint64_t(1) << i)) ? +1.0f : -1.0f);
		}
		// LHIST part (bits 0..11)
		for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
		{
			x_in[GLOBAL_HISTORY_LENGTH + j] =
			    ((s.lhist & (1u << j)) ? +1.0f : -1.0f);
		}
	}

	// ─────────────────────────────────────────────────────────────────────
	// Build the student’s feature vector x_stu[0..76]:
	// x_stu[0] = +1 (bias), x_stu[1..64] = +1/−1 from ghist, x_stu[65..76] = +1/−1 from lhist.
	inline void build_student_input(const TraceSample &s, int *x_stu)
	{
		x_stu[0] = +1; // bias
		// GHIST
		for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
		{
			x_stu[1 + i] = ((s.ghist & (uint64_t(1) << i)) ? +1 : -1);
		}
		// LHIST
		for (int j = 0; j < LOCAL_HISTORY_LENGTH; ++j)
		{
			x_stu[1 + GLOBAL_HISTORY_LENGTH + j] =
			    ((s.lhist & (1u << j)) ? +1 : -1);
		}
	}

}; // end class my_predictor

#endif // MY_PREDICTOR_H
