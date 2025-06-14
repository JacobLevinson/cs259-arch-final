// my_predictor.h
// This file contains a sample my_predictor class.
// It is a simple 32,768-entry gshare with a history length of 15.
// Note that this predictor doesn't use the whole 32 kilobytes available
// for the CBP-2 contest; it is just an example.

class my_update : public branch_update {
public:
	unsigned int index1; // Additional index variables required for multiple tables
	unsigned int index2;
	unsigned int index3;
};

class my_predictor : public branch_predictor {
public: // Table and history sizes optimized through fine tuning and experimentation
#define HISTORY_LENGTH1	15
#define TABLE_BITS1	22
#define HISTORY_LENGTH2	13
#define TABLE_BITS2	22
#define HISTORY_LENGTH3	14
#define TABLE_BITS3	22

	my_update u;
	branch_info bi;
	unsigned int history1;
	unsigned char tab1[1<<TABLE_BITS1];
	unsigned int history2;
	unsigned char tab2[1<<TABLE_BITS2];
	unsigned int history3;
	unsigned char tab3[1<<TABLE_BITS3];

	my_predictor (void) : history3(0), history2(0), history1(0) { // Constructor: Initialize all tables and histories
		memset (tab3, 0, sizeof (tab3));
		memset (tab2, 0, sizeof (tab2));
		memset (tab1, 0, sizeof (tab1));
	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.index1 = 
				  (history1 << (TABLE_BITS1 - HISTORY_LENGTH1)) 
				^ (b.address & ((1<<TABLE_BITS1 / 2)-1)); // Lower half hash function
			bool pred1 = tab1[u.index1] >> 1;
			u.index2 = 
				  (history2 << (TABLE_BITS2 - HISTORY_LENGTH2)) 
				^ (b.address & ((1<<TABLE_BITS2)-1)); // Full address hash function
			bool pred2 = tab2[u.index2] >> 1;
			u.index3 = 
				  (history3 << (TABLE_BITS3 - HISTORY_LENGTH3)) 
				^ (b.address & ((1<< (3*TABLE_BITS3 / 4))-1) << (TABLE_BITS3 / 4)); // Lower 3/4 hash function
			bool pred3 = tab3[u.index3] >> 1;

			int majority = (int) pred1 + (int) pred2 + (int) pred3; // Branch predictors vote
			if (majority >= 2) { // If majority predict true
				u.direction_prediction (true);
			}
			else { // Majority predicts false
				u.direction_prediction (false);
			}
			
		} else {
			u.direction_prediction (true);
		}
		u.target_prediction (0);
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		if (bi.br_flags & BR_CONDITIONAL) {
			unsigned char *c1 = &tab1[((my_update*)u)->index1]; // Update for predictor 1
			if (taken) {
				if (*c1 < 3) (*c1)++;
			} else {
				if (*c1 > 0) (*c1)--;
			}
			history1 <<= 1;
			history1 |= taken;
			history1 &= (1<<HISTORY_LENGTH1)-1;

			unsigned char *c2 = &tab2[((my_update*)u)->index2]; // Update for predictor 2
			if (taken) {
				if (*c2 < 3) (*c2)++;
			} else {
				if (*c2 > 0) (*c2)--;
			}
			history2 <<= 1;
			history2 |= taken;
			history2 &= (1<<HISTORY_LENGTH2)-1;

			unsigned char *c3 = &tab3[((my_update*)u)->index3]; // Update for predictor 3
			if (taken) {
				if (*c3 < 3) (*c3)++;
			} else {
				if (*c3 > 0) (*c3)--;
			}
			history3 <<= 1;
			history3 |= taken;
			history3 &= (1<<HISTORY_LENGTH3)-1;
		}
	}
};
