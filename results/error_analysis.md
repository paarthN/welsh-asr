# Error analysis

## Comparison

| Model | WER | CER | WER (no digits) | CER (no digits) |
|---|---|---|---|---|
| Whisper-small (zero-shot) | 0.5987 | 0.2289 | 0.5853 | 0.2149 |

## Worst failure cases (openai/whisper-small)

- **WER 8.65** (12.3s)
  - Reference: `dechreuodd diwylliannau a llwythau hynafol eu cadw ar gyfer mynediad hawdd at laeth gwallt cig a chrwyn`
  - Predicted: `maer ydych chin meddwl ar ydych chin meddwl ar ydych chin meddwl ar ydych chin meddwl ar ydych chin meddwl ar ydych chin meddwl ar ydych chin meddwl ar ydych ch`
- **WER 8.20** (19.3s)
  - Reference: `sut bynnag o ganlyniad ir sianeli cyfathrebu araf gallai ffasiynau yn y gorllewin fod ar ei hôl hi o hyd at 25 i 30 mlynedd`
  - Predicted: `sydd bynnag o gynlluniad ir sianeli cyfathrebi araf gallai ffasinnau yn y gorchewin fod ar ehol hi oi hyd at 20 5 i 30 000 000 000 000 000 000 000 000 000 000 0`
- **WER 6.17** (5.2s)
  - Reference: `yn union fel y maer lleuad yn tynnu ar y ddaear gan achosi llanwau felly maer llwybr llaethog yn bwrw grym ar alaeth sagittarius`
  - Predicted: `rwyf in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in meddwl in me`
- **WER 3.73** (18.3s)
  - Reference: `ailadeiladodd swltan morocor ddinas fel daru i badya a rhoddwyd yr enw casablanca iddi gan fasnachwyr sbaenaidd a sefydlodd ganolfannau masnachu yno`
  - Predicted: `y llywodraeth llywodraeth cymrur llywodraeth cymrur llywodraeth cymrur llywodraeth cymrur llywodraeth cymrur llywodraeth cymrur llywodraeth cymrur llywodraeth c`
- **WER 3.05** (18.1s)
  - Reference: `mae llawer o ddynion a menywod yn fyw o hyd a oroesodd eu cyfnod yma a llawer yn rhagor oedd ag anwyliaid a gafodd eu llofruddio neu eu gweithio i farwolaeth ym`
  - Predicted: `maer bwysig ar bwysig wedi cael ei wneud or hyfforddiadau ir bwysig ar bwysig ir bwysig ir bwysig ir bwysig ir bwysig ir bwysig ir bwysig ir bwysig ir bwysig ir`
