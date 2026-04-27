# AI Superintelligence Concept Demo
# Simulating a simple learning system

class SimpleAI:
    def __init__(self):
        self.knowledge = {}

    def learn(self, key, value):
        self.knowledge[key] = value

    def predict(self, key):
        return self.knowledge.get(key, "I don't know yet")

# Create AI
ai = SimpleAI()

# Learning phase
ai.learn("2+2", 4)
ai.learn("3+3", 6)

# Prediction phase
print("2+2 =", ai.predict("2+2"))
print("5+5 =", ai.predict("5+5"))
