class Model:
	def __init__(self, name, device):
		self.name = name
		self.device = device

	def create_model(self):
		pass

	def train(self, ds, num_epochs):
		pass

	def evaluate(self, dl):
		pass

	def save(self):
		pass
