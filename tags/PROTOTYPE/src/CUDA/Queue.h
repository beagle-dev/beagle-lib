/*
 *  Queue.h
 *  BEAGLE
 *
 * @author Marc Suchard
 *
 */

#define QUEUESIZE       1000

class Queue {
private:
	int q[QUEUESIZE+1]; /* body of queue */
	int first; /* position of first element */
	int last; /* position of last element */
	int count; /* number of queue elements */

public:
	void initQueue();
	void enQueue(int x);
	int deQueue();
	int queueEmpty();
	void printQueue();
};

