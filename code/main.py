import pandas as pd
import torch
import math
from argparse import ArgumentParser
from datasets import load_dataset,Dataset, DatasetDict
from transformers import GPT2Tokenizer,DataCollatorForLanguageModeling,GPT2LMHeadModel, TrainingArguments, Trainer, GPT2Config,GPT2ForSequenceClassification,DataCollatorWithPadding
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import pickle

def train_model(data_path, save_model_path):

    ######pre-train######

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Assuming 'tokenizer' is your GPT-2 tokenizer
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

    def tokenize_function(examples):
        # Tokenize the text and prepare 'labels' for CLM
        tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        # For CLM, labels are the same as input_ids but shifted so every token is predicted
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Create a configuration object
    config = GPT2Config(
        vocab_size=50257,  # standard GPT-2 vocabulary size
        n_positions=1024,  # max sequence length; consider reducing for even less memory usage
        n_ctx=1024,  # context size; usually the same as n_positions
        n_embd=1024,  # smaller model dimensionality
        n_layer=1,  # fewer layers
        n_head=8,  # fewer attention heads
        use_cache=False,  # Disable caching to reduce memory usage further
    )

    model = GPT2LMHeadModel(config)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,  # Lowered batch size
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity: {perplexity}")

    # Save the model and tokenizer
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)

    ######fine-tuning######
    def tokenize_function(examples):
        return tokenizer(examples["utterances"], truncation=True, max_length=128, padding="max_length")
    df = pd.read_csv(data_path)
    df=df.sample(frac = 1,random_state=42)
    df=df.fillna('none')
    def tokenize_text(examples):
        return tokenizer(examples["utterances"], truncation=True)

    def compute_metrics(p):
        predictions = p.predictions > 0.5  # Assuming threshold for classification is 0.5, adjust as needed
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1': f1,}

    def one_hot_encode(text, token_to_index):
        one_hot_vector = [0 for i in range(len(token_to_index))]

        for token in str(text) if isinstance(text, float) else text.split(' '):
            if token!='none':
                one_hot_vector[token_to_index[token]] = 1
        return one_hot_vector

    tokenizer = GPT2Tokenizer.from_pretrained(save_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    # Create Vocab list
    counter = Counter(set([val for row in df['utterances'].to_list() for val in row.split(' ')]))
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    unk_token = '<unk>'
    default_index = 5000
    tokens = list(set([val for row in df['utterances'].to_list() for val in row.split(' ')]))
    # Create Relations list
    core_rels_list = sorted(list(set([j for i in df["Core Relations"].to_list() for j in str(i).split(" ")])))
    core_rels_list.remove('none')
    core_rels_to_index_dict = {val: index for index, val in enumerate(sorted(core_rels_list))}
    index_to_core_rels_dict = {index: val for val, index in core_rels_to_index_dict.items()}

    model = GPT2ForSequenceClassification.from_pretrained(
        save_model_path,
        num_labels=len(core_rels_to_index_dict),
        problem_type="multi_label_classification",
    )
    df['core_rel_vectorized'] = df['Core Relations'].apply(lambda x: one_hot_encode(x, core_rels_to_index_dict))
    df.drop(columns=['IOB Slot tags', 'Core Relations'], inplace=True)
    df.rename(columns={
        'core_rel_vectorized':'labels'
    },inplace=True)
    train_ds = Dataset.from_pandas(df)
    train_ds = train_ds.map(tokenize_function, batched=True)
    train_ds, valid_ds = train_ds.train_test_split(test_size=0.2).values()
    ds = DatasetDict()
    ds['train'] = train_ds
    ds['valid'] = valid_ds
    # cast label IDs to floats
    ds.set_format("torch")
    ds = (ds.map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]).rename_column("float_labels", "labels"))
    ds  = ds.map(tokenize_text, batched=True)
    model.config.pad_token_id = model.config.eos_token_id
    training_args = TrainingArguments(
        "testtraining", 
        num_train_epochs=50,
        learning_rate=1e-4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds['valid'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    evaluate_result=trainer.evaluate()
    print(evaluate_result)
    # Save variables for test
    my_data = {'core_rels_to_index_dict': core_rels_to_index_dict,'index_to_core_rels_dict':index_to_core_rels_dict}
    with open('./my_saved_data.pkl', 'wb') as file:
        pickle.dump(my_data, file)
        
    # Save the model and tokenizer
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)


def test_model(data_path, model_path, output_path):
    # ... do data processing/prediction generation here
    with open('my_saved_data.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    core_rels_to_index_dict=loaded_data['core_rels_to_index_dict']
    index_to_core_rels_dict=loaded_data['index_to_core_rels_dict']
    test_data = pd.read_csv(data_path)
    model = GPT2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(core_rels_to_index_dict),
        problem_type="multi_label_classification",
    )
    tokenizer = GPT2Tokenizer.from_pretrained('./pre_trained_gpt2_model')
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["utterances"], truncation=True, max_length=128, padding="max_length")
    def tokenize_text(examples):
        return tokenizer(examples["utterances"], truncation=True)

    def compute_metrics(p):
        predictions = p.predictions > 0.5  # Assuming threshold for classification is 0.5, adjust as needed
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1': f1,}

    test_df = test_data[['utterances']]
    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.map(tokenize_function, batched=True)
    ds = DatasetDict()

    ds.set_format("torch")
    ds = (ds.map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]).rename_column("float_labels", "labels"))

    ds['test'] = test_ds
    ds  = ds.map(tokenize_text, batched=True)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(ds['test'])
    # save the predictions as a CSV
    transfer_test=predictions[0]
    predicted_rel=[]
    for i in transfer_test:
        transfer_test_thresh=torch.sigmoid(torch.from_numpy(i))>0.5
        predicted_rel_sentence=[]
        for i in range(len(transfer_test_thresh)-1):
            if transfer_test_thresh[i].item()==True:
                predicted_rel_sentence.append(index_to_core_rels_dict[i])
        predicted_rel.append(predicted_rel_sentence)
    for i in predicted_rel:
        if 'none' in i:
            i.remove('none')
    # df_output= pd.read_csv(data_path)
    df_output= pd.read_csv('./hw1_test.csv')
    df_output['Core Relations'] = [' '.join(map(str, sublist)) if isinstance(sublist, list) else str(sublist) for sublist in predicted_rel]
    # # Write the modified DataFrame back to the CSV file
    df_output.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser("homework CLI")

    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")

    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    if args.train:
        train_model(args.data, args.save_model)
    if args.test:
        test_model(args.data, args.model_path, args.output)