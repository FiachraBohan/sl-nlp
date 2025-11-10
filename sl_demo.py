import classla
from transformers import AutoModelForCausalLM, AutoTokenizer


classla.download('sl')
nlp = classla.Pipeline('sl')

doc = nlp("France Prešeren je rojen v Vrbi.")
print(doc.to_conll())

tokenizer = AutoTokenizer.from_pretrained("sambanovasystems/SambaLingo-Slovenian-Base")
model = AutoModelForCausalLM.from_pretrained("sambanovasystems/SambaLingo-Slovenian-Base", device_map="auto", dtype="auto",offload_folder="./offload")
print("Model loaded:", model.config._name_or_path)
