Basline: MobileVit -> mean over all slices for final prediction
2. Step: MobileVit -> weighted aggregation over slices -> more interpretable
3. Step: Transformer  -> mean & weighted
4. Step: NN for aggregation
5. Step: Self Distillation


MobileVit usage example:
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-x-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-x-small")

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
