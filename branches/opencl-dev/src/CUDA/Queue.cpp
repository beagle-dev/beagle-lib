/*
 *  Queue.cpp
 *  BEAGLE
 *
 * @author Marc Suchard
 *
 */

#include <stdio.h>
#include "Queue.h"

void Queue::initQueue() {
	first = 0;
	last = QUEUESIZE - 1;
	count = 0;
}

void Queue::enQueue(int x) {
	if (count >= QUEUESIZE)
		printf("Warning: queue overflow enqueue x=%d\n", x);
	else {
		last = (last + 1) % QUEUESIZE;
		q[last] = x;
		count = count + 1;
	}
}

int Queue::deQueue() {
	int x;

	if (count <= 0)
		fprintf(stderr,"Warning: empty queue dequeue.\n");
	else {
		x = q[first];
		first = (first + 1) % QUEUESIZE;
		count = count - 1;
	}

	return (x);
}

int Queue::queueEmpty() {
	if (count <= 0)
		return 1;
	else
		return 0;
}

void Queue::printQueue() {
	int i, j;

	i = first;

	while (i != last) {
		printf("%d ", q[i]);
		i = (i + 1) % QUEUESIZE;
	}
	printf("%d ", q[i]);
	printf("\n");
}
