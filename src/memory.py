import random
import numpy as np

class MemoryReplay(object):

	def __init__(self,
				 max_size=10000, # memory size
				 bs=64,          # batch size
				 im_size=84,     # image dimension
				 stack=4):       # number of stacked frames

		self.s = np.zeros((max_size, stack+1, im_size, im_size), dtype=np.float32)
		self.r = np.zeros(max_size, dtype=np.float32)
		self.a = np.zeros(max_size, dtype=np.int32)
		#self.ss = np.zeros_like(self.s)
		self.done = np.array([True]*max_size)

		self.max_size = max_size
		self.bs = bs
		self._cursor = 0
		self.total_idx = list(range(self.max_size))


	def put(self, sras):

		self.s[self._cursor] = sras[0]
		self.a[self._cursor] = sras[1]
		self.r[self._cursor] = sras[2]
		#self.ss[self._cursor] = sras[3]
		self.done[self._cursor] = sras[3]

        self._cursor += 1
        if self._cursor == self.max_size:
            self._cursor = 0


	def batch(self):

		sample_idx = random.sample(self.total_idx, self.bs)
		s = self.s[sample_idx, :4]
		a = self.a[sample_idx]
		r = self.r[sample_idx]
		#ss = self.ss[sample_idx]
		ss = self.s[sample_idx, 1:]
		done = self.done[sample_idx]

		return s, a, r, ss, done
