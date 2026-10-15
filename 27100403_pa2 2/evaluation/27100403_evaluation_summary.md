# Results summary

## Training runs

| run              | steps | lr    | beta | bs x accum | val loss | val reward acc | val reward margin | wall clock |
|------------------|-------|-------|------|------------|----------|----------------|-------------------|------------|
| sft              | 937   | 2e-05 | -    | 2x16       | 1.372    | -              | -                 | 0h06m41s   |
| dpo_hh           | 468   | 1e-06 | 0.1  | 1x32       | 0.7103   | 0.55           | 0.0622            | 1h43m25s   |
| dpo_uf           | 468   | 1e-06 | 0.1  | 1x32       | 0.5907   | 0.69           | 0.511             | 2h32m27s   |
| lora_dpo_persona | -     | 5e-05 | 0.1  | 2x4        | -        | -              | -                 | 0h03m18s   |

## Pairwise judging

| comparison           | model                         | n   | wins | losses | ties | win rate (w/ ties) | win rate (excl. ties) | failed | judge                   |
|----------------------|-------------------------------|-----|------|--------|------|--------------------|-----------------------|--------|-------------------------|
| dpo_hh_vs_sft        | qwen2.5-0.5b-dpo-hh           | 300 | 194  | 84     | 22   | 0.6467             | 0.6978                | 0      | google/gemini-3.6-flash |
| dpo_hh_vs_sft        | qwen2.5-0.5b-sft              | 300 | 84   | 194    | 22   | 0.28               | 0.3022                | 0      | google/gemini-3.6-flash |
| dpo_uf_vs_sft        | qwen2.5-0.5b-dpo-uf           | 300 | 113  | 134    | 53   | 0.3767             | 0.4575                | 0      | google/gemini-3.6-flash |
| dpo_uf_vs_sft        | qwen2.5-0.5b-sft              | 300 | 134  | 113    | 53   | 0.4467             | 0.5425                | 0      | google/gemini-3.6-flash |
| lora_persona_vs_base | smollm2-360m-instruct         | 26  | 21   | 5      | 0    | 0.8077             | 0.8077                | 0      | google/gemini-3.6-flash |
| lora_persona_vs_base | smollm2-360m-lora-dpo-persona | 26  | 5    | 21     | 0    | 0.1923             | 0.1923                | 0      | google/gemini-3.6-flash |
| pilot                | qwen2.5-0.5b-dpo-uf           | 10  | 3    | 6      | 1    | 0.3                | 0.3333                | 0      | google/gemini-3.6-flash |
| pilot                | qwen2.5-0.5b-sft              | 10  | 6    | 3      | 1    | 0.6                | 0.6667                | 0      | google/gemini-3.6-flash |
