import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ChatBrew"
print(f"{'Welcome to chat with ChatBrew':^60}\n\n")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])  # Ensure it is a 2D tensor (1, num_features)
    X = torch.from_numpy(X).float()  # Fix: Convert NumPy array to tensor and set dtype to float

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I'm not sure I understand what you mean by ")


#         1. General Inquiries
#         You: "Hi ChatBrew, what can I do here?"
#         You: "Can you tell me about the coffee you serve?"
#         You: "What’s on the menu today?"
#         You: "What drinks do you offer?"
#         You: "Do you have any pastries?"
#         2. Ordering Process
#         You: "How do I place an order?"
#         You: "Can I order coffee through the app?"
#         You: "Can I place an order for delivery?"
#         You: "How can I order my coffee?"
#         You: "Do you offer any customization for my drinks?"
#         3. Payment & Delivery
#         You: "What payment methods do you accept?"
#         You: "Can I pay with PayPal?"
#         You: "How long will it take to deliver my coffee?"
#         You: "How soon can I get my coffee delivered?"
#         4. Loyalty Program & Offers
#         You: "Do you have a loyalty program?"
#         You: "How can I earn rewards?"
#         You: "Do you offer any discounts for regular customers?"
#         You: "Can I use my points for a free coffee?"
#         5. Shop Information
#         You: "What are your working hours?"
#         You: "Where are you located?"
#         You: "How can I find your coffee shop?"
#         6. Fun & Engagement
#         You: "Tell me a joke!"
#         You: "Can you make me laugh?"
#         You: "What’s the best way to enjoy a cup of coffee?"
#         You: "What’s your favorite type of coffee?"
#         7. Customization
#         You: "Can I get a latte with oat milk?"
#         You: "What milk options do you offer?"
#         You: "Do you have any sugar-free syrups?"
#         You: "Can I add an extra shot to my espresso?"
#         8. Status of Order
#         You: "Where is my order?"
#         You: "Has my order shipped?"
#         You: "When will my coffee be ready?"
#         You: "How do I track my order?"
#         9. Specific Coffee Types
#         You: "Do you serve iced coffee?"
#         You: "What’s the difference between a cappuccino and a latte?"
#         You: "Do you offer cold brew?"
#         You: "What type of espresso drinks do you serve?"
#         10. Coffee Shop’s Special Features
#         You: "What makes your coffee shop unique?"
#         You: "Can I customize my coffee order?"
#         You: "Do you have vegan-friendly options?"
#         You: "Can I order for pickup?"


# Hi
#
# what can I do here?
#
# What’s on the menu today?
#
# How do I place an order?
#
# Can I pay with PayPal?
#
# Do you offer any discounts for regular customers?
#
# How can I find your coffee shop?
#
# Can I add an extra shot to my espresso?
#
# What type of espresso drinks do you serve?





# Hi

# How can I order my coffee? 

# What payment methods do you accept? 
# what paument methods do you accept?

# How can I earn rewards?
# how can i earn rewards?

# Can I customize my coffee order?

# Do you serve iced coffee?

# Can I customize my coffee order?"
