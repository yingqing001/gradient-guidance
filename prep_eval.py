import requests

# URL to the ImageNet class list (example)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Fetch the list
response = requests.get(url)
imagenet_classes = response.text.splitlines()

# Write the list to a local file
with open("imagenet_classes.txt", "w") as file:
    for class_name in imagenet_classes:
        file.write(class_name + "\n")

# Print the first 20 classes as a sample
for i, class_name in enumerate(imagenet_classes[:20]):
    print(f"{i+1}: {class_name}")
