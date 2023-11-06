import stow
from configs import ModelConfigs

dataset = []
vocab = set()
max_len = 0

for file in stow.ls(stow.join("../dataset", "../preprocessed_dataset")):
    dataset.append([stow.relpath(str(file)), file.name])
    vocab.update(list(file.name))
    max_len = max(max_len, len(file.name))

configs = ModelConfigs()

configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()
