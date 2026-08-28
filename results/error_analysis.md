# Error analysis

## Comparison

| Model | WER | CER | WER (no digits) | CER (no digits) |
|---|---|---|---|---|
| Whisper-small (zero-shot) | 0.5987 | 0.2289 | 0.5853 | 0.2149 |
| XLS-R 300m (fine-tuned) | 0.3991 | 0.1141 | 0.3879 | 0.1098 |
| **relative change** | -33.3% | -50.1% | -33.7% | -48.9% |

## Worst failure cases (pnawani/welsh-asr-xlsr-300m)

- **WER 1.00** (5.2s)
  - Reference: `yn union fel y maer lleuad yn tynnu ar y ddaear gan achosi llanwau felly maer llwybr llaethog yn bwrw grym ar alaeth sagittarius`
  - Predicted: ``
- **WER 1.00** (1.5s)
  - Reference: `maen gysylltiedig â ond nid yn cynnwys teithio ar sgiau neu fynydda alpaidd y rhai olaf ar dirwedd serth ac yn gofyn am sgiau ac esgidiau llawer cryfach`
  - Predicted: ``
- **WER 1.00** (12.3s)
  - Reference: `dechreuodd diwylliannau a llwythau hynafol eu cadw ar gyfer mynediad hawdd at laeth gwallt cig a chrwyn`
  - Predicted: `aisiant gour goers en trifus fudant o bwyzefydim fo iz arsis tu mioghe minta sgins`
- **WER 0.98** (33.1s)
  - Reference: `nid yw eu hymddygiad thermol mor sefydlog ag ogofâu mawr ar y ddaear syn aml yn cynnal tymheredd eithaf cyson ond maen gyson â bod yn dyllau dwfn yn y ddaear me`
  - Predicted: `jdemalfhywio isnodaus difodaz laj cewzon od dad offtwmn men dim fely gonstenten gwichers fod i dys gonsistan twyds vind dib phos indysedglewn gosin oei lodir gw`
- **WER 0.98** (18.1s)
  - Reference: `mae llawer o ddynion a menywod yn fyw o hyd a oroesodd eu cyfnod yma a llawer yn rhagor oedd ag anwyliaid a gafodd eu llofruddio neu eu gweithio i farwolaeth ym`
  - Predicted: `mini men a wy men astu aleich wsovafidiat den hy an me mo adlygg iwns hwermoedad o putoydat hi buots noyns an nongoeus`
