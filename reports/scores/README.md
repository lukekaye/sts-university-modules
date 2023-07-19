In similarity_evaluation_results.csv, the listed scores are ordered by the following models:

- Longformer with no fine-tuning
- BigBird with no fine-tuning
- DistilRoBERTa with no fine-tuning
- DistilRoBERTa fine-tuned on general STS data
- Longformer fine-tuned on our data by SimCSE
- Longformer fine-tuned on our data by CT
- BigBird fine-tuned on our data by SimCSE
- BigBird fine-tuned on our data by CT
- BigBird fine-tuned on our data by TSDAE
- DistilRoBERTa fine-tuned on our data by SimCSE
- DistilRoBERTa fine-tuned on our data by CT
- DistilRoBERTa fine-tuned on our data by TSDAE
- DistilRoBERTa fine-tuned on general STS data, then fine-tuned on our data by SimCSE
- DistilRoBERTa fine-tuned on general STS data, then fine-tuned on our data by CT
- DistilRoBERTa fine-tuned on general STS data, then fine-tuned on our data by TSDAE

the fine-tuning approach used for DistilRoBERTa on general STS data can be found at: https://huggingface.co/sentence-transformers/all-distilroberta-v1