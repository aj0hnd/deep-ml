class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		 # Implement addition here
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, (self, other), '+')
		def _backward():
			self.grad += 1.0 * out.grad
			other.grad += 1.0 * out.grad
		
		out._backward = _backward
		return out

	def __mul__(self, other):
		# Implement multiplication here
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, (self, other), '*')
		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		
		out._backward = _backward
		return out

	def relu(self):
		# Implement ReLU here
		out = Value(max(0, self.data), (self,), 'relu')
		def _backward():
			self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
		
		out._backward = _backward
		return out

	def backward(self):
		# Implement backward pass here
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)

		build_topo(self)

		self.grad = 1
		for node in reversed(topo):
			node._backward()