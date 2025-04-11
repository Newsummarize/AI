from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

modelName = "EbanLee/kobart-summary-v3"

tokenizer = PreTrainedTokenizerFast.from_pretrained(modelName)
model = BartForConditionalGeneration.from_pretrained(modelName)

def summarize_text(text, max_length=200, min_length=20):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)

    summary_text_ids = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=1.0,
    max_length=max_length,
    min_length=min_length,
    num_beams=6,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3,
    )

    return tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)