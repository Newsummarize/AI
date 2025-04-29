from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import re

modelName = "EbanLee/kobart-summary-v3"
tokenizer = PreTrainedTokenizerFast.from_pretrained(modelName)
model = BartForConditionalGeneration.from_pretrained(modelName)

def summarize_text(text):
    inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=1026
    )

    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=0.7,
        max_length=130,
        min_length=12,
        num_beams=4,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    if len(summary) > 170:
        end_match = re.search(r'[.!?](?=[^.!?]*$)', summary[:200])
        if end_match:
            return summary[:end_match.end()]
        return summary[:170]
    
    return summary
