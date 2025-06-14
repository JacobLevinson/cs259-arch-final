// my_predictor.h
/*
  Perceptron branch predictor (global-plus-local):

  1.  Start with the bias weight W0.
      Add one term for every bit in the *global* history buffer.
      Add another small set of terms for the branch’s own private
      *local* history register (last 8 outcomes of this PC).

	output = W0
	       + WG1 * gHist[0]   + … + WG64 * gHist[63]
	       + WL1 * lHist[0]   + … + WL8  * lHist[7]

      A history bit is +1 when the earlier branch was taken, −1 when
      it was not-taken.

  2.  If the final sum is positive we predict “taken”.
      If it is negative we predict “not-taken”.
      A big magnitude means we’re confident; a value close to zero
      means we’re just guessing.

  3.  We train only when the prediction was wrong *or* when the sum
      sat too close to zero to be trustworthy.
      Every weight that participated moves by a single step toward the
      correct direction, then gets clamped so it always stays inside the
      signed 8-bit range [-128…127].

  4.  The table is indexed with the low 12 bits of the branch address,
      so each static branch has its own row of weights.  All rows share
      the global history, but each row also keeps its own 8-bit local
      history shift-register.
*/

class my_update : public branch_update
{
public:
	unsigned int index; // which weight row we touched
	int output;	    // raw perceptron sum
};

class my_predictor : public branch_predictor
{
public:
// ─ Tunable parameters
#define GLOBAL_HISTORY_LENGTH 64 // length of shared global history
#define LOCAL_HISTORY_LENGTH 12	 // small private history per static branch
#define TABLE_BITS 16		 // 2^16 = 65,536 rows
#define THRESHOLD 140		 // confidence margin
#define WEIGHT_MAX 127		 // 8-bit signed saturation
#define WEIGHT_MIN -128
	// Internal storage
	my_update u;
	branch_info bi;

	int ghist[GLOBAL_HISTORY_LENGTH];			   // global history
	int g_weights[1 << TABLE_BITS][GLOBAL_HISTORY_LENGTH + 1]; // +1 for bias

	unsigned char lhist[1 << TABLE_BITS];		      // 8-bit shift-reg
	int l_weights[1 << TABLE_BITS][LOCAL_HISTORY_LENGTH]; // local weights

	// Stats
	unsigned long long total_predictions = 0;
	unsigned long long total_updates = 0;
	unsigned long long weak_predictions = 0;
	unsigned long long strong_correct = 0;
	unsigned long long strong_wrong = 0;

	my_predictor()
	{
		memset(ghist, 0, sizeof(ghist));
		memset(g_weights, 0, sizeof(g_weights));
		memset(lhist, 0, sizeof(lhist));
		memset(l_weights, 0, sizeof(l_weights));
	}

	~my_predictor()
	{
		FILE *f = fopen("perceptron_stats.txt", "a");
		if (!f)
			return;
		fprintf(f,
			"\n=== Perceptron Predictor Statistics ===\n"
			"Total Predictions: %llu\n"
			"Total Updates (training events): %llu\n"
			"Weak Predictions (|output| <= threshold): %llu\n"
			"Strong Correct Predictions: %llu\n"
			"Strong Wrong Predictions: %llu\n"
			"========================================\n",
			total_predictions, total_updates, weak_predictions,
			strong_correct, strong_wrong);
		fclose(f);
	}

	// make a direction prediction for every branch
	branch_update *predict(branch_info &b)
	{
		bi = b;

		if (b.br_flags & BR_CONDITIONAL)
		{
			// Hash in a couple of very recent branch outcomes into the raw PC bits so that two different
			// static branches that share the same low 13 bits of address don’t always collide in the weight table.
			// Basically, free extra storage

			/*  Fold four reasonably independent history bits into the raw PC.
			We shift each bit by a different amount so the information lands in
			otherwise-unused positions of the index word, then let the final mask
			pick TABLE_BITS of them.
			- ghist[1]  → recent    (very correlated)
			- ghist[2]  → ”         (very correlated)
			- ghist[5]  → slightly older
			- ghist[13] → much older, breaks up long patterns
			*/
			unsigned idx = b.address	   // low PC bits
				       ^ (ghist[1] << 5)   // inject bit #1
				       ^ (ghist[2] << 9)   // inject bit #2
				       ^ (ghist[5] << 12)  // inject bit #5
				       ^ (ghist[13] << 2); // inject bit #13

			u.index = idx & ((1 << TABLE_BITS) - 1); // keep TABLE_BITS bits

			// Start with bias
			int sum = g_weights[u.index][0];

			// Global history contribution
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
				sum += g_weights[u.index][i + 1] * (ghist[i] ? 1 : -1);

			// Local history contribution (8 most-recent outcomes of *this* PC)
			unsigned char lh = lhist[u.index];
			for (int i = 0; i < LOCAL_HISTORY_LENGTH; ++i)
				sum += l_weights[u.index][i] * ((lh & (1u << i)) ? 1 : -1);

			u.output = sum;
			u.direction_prediction(sum >= 0);

			++total_predictions;
			if (abs(sum) <= THRESHOLD)
				++weak_predictions;
		}
		else
		{
			u.direction_prediction(true); // ignore non-conditional branches
		}

		u.target_prediction(0);
		return &u;
	}

	// train the perceptron after the actual outcome is known
	void update(branch_update *buf, bool taken, unsigned /*target*/)
	{
		if (!(bi.br_flags & BR_CONDITIONAL))
			return;

		auto *mu = static_cast<my_update *>(buf);
		bool correct = ((mu->output >= 0) == taken);
		bool strong = (abs(mu->output) > THRESHOLD);
		if (strong)
			(correct ? ++strong_correct : ++strong_wrong);

		int t = taken ? 1 : -1;

		// Train if needed
		if (!correct || !strong)
		{
			++total_updates;
			int *gw = g_weights[mu->index];
			unsigned char lh = lhist[mu->index];
			int *lw = l_weights[mu->index];

			// bias
			gw[0] = saturate(gw[0] + t);

			// global weights
			for (int i = 0; i < GLOBAL_HISTORY_LENGTH; ++i)
				gw[i + 1] = saturate(gw[i + 1] + t * (ghist[i] ? 1 : -1));

			// local weights
			for (int i = 0; i < LOCAL_HISTORY_LENGTH; ++i)
				lw[i] = saturate(lw[i] + t * ((lh & (1u << i)) ? 1 : -1));
		}

		// Update global history shift-register
		for (int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; --i)
			ghist[i] = ghist[i - 1];
		ghist[0] = taken ? 1 : 0;

		// Update local history shift-register for this PC
		lhist[mu->index] = ((lhist[mu->index] << 1) | (taken ? 1 : 0)) &
				   ((1u << LOCAL_HISTORY_LENGTH) - 1);
	}

private:
	static int saturate(int x)
	{
		if (x > WEIGHT_MAX)
			return WEIGHT_MAX;
		if (x < WEIGHT_MIN)
			return WEIGHT_MIN;
		return x;
	}
};
