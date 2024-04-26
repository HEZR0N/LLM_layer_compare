# LLM_layer_compare

This repo was created for an assignment in my NLP/LLM class. The goal was to a fine-tuned model and examine the differences in the probability distibutions for different layers in the model.
The finetuned model was based off of:
 - https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main
## Requirements
 - Python 3.8

Required Libraries/Modules made be installed with this command:
```
pip install transformers trl rouge bert_score evaluate nltk
```
## Usage
Edit `layer_compare.py` to set the `model_path` variable to the path to the model your using and set `access_token` to the value of your huggingface access token if you're using it:
```
# load model and tokenizer
access_token="YOUR_HF_TOKEN"
model_path = "model/path"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
```
Then you can run the following command:     
`python layer_compare.py`    

The program is currently set to generate ouputs (one word text completions) for all data in the dataset, but only creates plots and evalutes metrics for the layers of the first example.

Below are the plots, metrics, and results that the program will output.

## Example Input
Input: "Birds fly high in the "       
Expected Output: "sky"

## Plots

![image](https://github.com/HEZR0N/LLM_layer_compare/assets/99786488/7b3cce3c-b18c-41e5-aedf-b1900acced50)       

![image](https://github.com/HEZR0N/LLM_layer_compare/assets/99786488/ae0f8eed-1439-4feb-acda-36ae5498ca22)         

![image](https://github.com/HEZR0N/LLM_layer_compare/assets/99786488/0197b251-f927-4067-8038-dca5d4586ae5)      

![image](https://github.com/HEZR0N/LLM_layer_compare/assets/99786488/f4f59615-136f-41f3-972a-0fb782f1c483)     


## Additional Output 
```
Layer  0  best prob:  3.131121411570348e-05 , best token:  elli
Layer  1  best prob:  3.14411154249683e-05 , best token:  Â
Layer  2  best prob:  3.152755743940361e-05 , best token:  ­
Layer  3  best prob:  3.161425411235541e-05 , best token:  bewerken
Layer  4  best prob:  3.1793351809028536e-05 , best token:  rok
Layer  5  best prob:  3.1955107260728255e-05 , best token:  ­
Layer  6  best prob:  3.229461071896367e-05 , best token:  ­
Layer  7  best prob:  3.2809482945594937e-05 , best token:  ­
Layer  8  best prob:  3.281622412032448e-05 , best token:  ­
Layer  9  best prob:  3.2842308428371325e-05 , best token:  ־
Layer  10  best prob:  3.29611539200414e-05 , best token:  ­
Layer  11  best prob:  3.345026925671846e-05 , best token:  ­
Layer  12  best prob:  3.3497242839075625e-05 , best token:  ­
Layer  13  best prob:  3.390374331502244e-05 , best token:  ­
Layer  14  best prob:  3.409220516914502e-05 , best token:  ­
Layer  15  best prob:  3.4438420698279515e-05 , best token:  Mär
Layer  16  best prob:  3.515127536957152e-05 , best token:  Mär
Layer  17  best prob:  3.608597398851998e-05 , best token:  Mär
Layer  18  best prob:  3.694871338666417e-05 , best token:  Mär
Layer  19  best prob:  3.766633017221466e-05 , best token:  Mär
Layer  20  best prob:  3.9184615161502734e-05 , best token:  Mär
Layer  21  best prob:  4.046218600706197e-05 , best token:  Fuß
Layer  22  best prob:  4.349834853201173e-05 , best token:  Fuß
Layer  23  best prob:  4.724090831587091e-05 , best token:  Fuß
Layer  24  best prob:  4.9987480451818556e-05 , best token:  Fuß
Layer  25  best prob:  5.092922947369516e-05 , best token:  Fuß
Layer  26  best prob:  5.5604788940399885e-05 , best token:  sky
Layer  27  best prob:  5.592667730525136e-05 , best token:  sky
Layer  28  best prob:  5.7451092288829386e-05 , best token:  sky
Layer  29  best prob:  6.332020711852238e-05 , best token:  sky
Layer  30  best prob:  9.181249333778396e-05 , best token:  sky
Layer  31  best prob:  0.00010334287799196318 , best token:  sky
Layer N: 1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32
Layer 1: elli	acknow	rix	irt	gover	BF	iple	ouver	川	wy	scri	cho	走	vens	somehow	steller	selves	eller	LL	scription	员	transferred	ellers	itle	tym	ethod	Â	lä	comm	хо	PRIV	ort
Layer 2: Â	­	־	Ã	ゃ	弟	klik	bewerken	ٰ	œuv	ٱ		varied	ŭ	rog	♡	ھ	dici	Fuß	technological	släktet	ẩ	plaat	prv	rå	KERN	ℓ	‑	ỹ	human	rcu	ær
Layer 3: ­	Â	Ã	♡	_**	För	‑	grp	TDM	expl	bewerken	ŭ	democr	{@	technological	intended	ゃ	atr	‐	human	oll	Corn	scientific	hollow	־	senses	''	ٰ	th	[…]	pop	Ces
Layer 4: bewerken	־	­	ゃ	ٰ	plaat	Â	ниш	ť	rå	‑	➤	ẓ	ŭ	kennis	ṯ	ķ	bekan	▸	Fuß	"*	振	̂	technological		ņ	‐	klik	랜	∂	scientific	préc
	іб	ẓ	éc	랜	bewerken	﻿	➤	ця	*"	краї	"*	EFF	grp	dispon	wahl	Lebens	Dum	klik	eron	fy
Layer 6: ­	‑	Â	'	-	pre	‐	interest	стре	Roman	Dum	fin	�	eff	various	Ã	aters	fier	interests	Amer	containing	rok	arms	scientific	rieb	frame	sn	era	multiple	sn	rend	Virgin
Layer 7: ­	‑	uv	aron	Â	riz	various	ses	‐	Ã	'	fare	interior	[…]	fin	vt	*)&	rieb	Haus	atr	ischer	geme	fier	orie	ák	Dum	dri	fun	tune	regard	�	>/
Layer 8: ­	‑	‐	riz	interior	Â	Ã	Nord	Haus	'	-,	high	nord	-	ara	[…]	erial	aron	senses	*)&	early	air	neck	''	�	fun	Dak	depth	aml	midst	aid	dri
Layer 9: ­	Ã	‑	ara	‐	Â	*)&	ã	Lil	LOB	nord	scribed	tow	[…]	klass	sole	rt	ived	clutch	�	EXPECT	Ludwig	ín	ismo	ŕ	̂	cio	rå	battles	Hav	geme	tune
Layer 10: ־	­	kennis	̂	лове	дри	EXPECT	ķ	scriptstyle	Ť	plaat	Fuß	ara	klik	giornata	ĕ	erh	ẓ	Lebens	Ď	гре	bewerken	dici	seh	klass	représ	‐	‑	̇	LOB	schen	ņ
Layer 11: ­	ť	EXPECT	LOB	elev	ara	־	erh	ẓ	klass	hö	̂	buch	Geme	ín	дри	Ď	schen	Fußball	Dum	klik	Lebens	PRESS	above	bewerken	ゃ	height	erial	кла	atr	ớ	лове
Layer 12: ­	klass	EXPECT	ť	LOB	giornata	‐	Mär	Â	erial	API	Como	ín	Lebens	Kö	klik	ara	Geme	bewerken	kennis	schen	ĕ	ớ	erh	Egy	־	comun	Fußball	PIN	PRESS	elev	‑
Layer 13: ­	LOB	giornata	Mär	EXPECT	För	Kö	PRESS	Â	klik	ť	̥	klass	zaj	̄	kennis	Egy	/******/	biologie	־	noten	bewerken	след	dirig	Geme	ḷ	Lebens	‐	Hö	ई	Zu	gioc
Layer 14: ­	/******/	Â	Mär	‐	klass	giornata	zaj	EXPECT	dirig	För	nitt	̥	Egy	̆	[…]	PRESS	soort	pierws	ḷ	DAI	Parte	Ď	hci	Kö	hoof	gioc	bewerken	vma	klik	vale	‑
Layer 15: ­	Mär	För	kennis	klass	giornata	klik	ई	zaj	Â	Musik	ḩ	erh	vess	кла	Egy	庆	dirig	.<	typen	áll	̆	▸	‐	DAI	LOB	➤	且	bewerken	Kön	KERN	há
Layer 16: Mär	klass	kennis	giornata	­	klik	ई	För	̣	typen	־	➤	Kön	ḷ	áll	vess	ľ	mű	soort	Ħ	ỹ	klub	贵	кла	Musik	Â	Ζ	zaj	‐	bewerken	►	kör
Layer 17: Mär	­	klass	ई	klik	ã	För	Musik	Â	giornata	klass	Fuß	animated	dirig	opening	zaj	biologie	há	кла	Zent	▸	іб	vess	贵	Egy	áll	osc	oka	̣	Ď	бор	atr
Layer 18: Mär	biologie	Fuß	klik	ٹ	Ζ	ई	ниш	vess	іб	netdev	Musik	VENDOR	ھ	há	Украї	Kön	▸	För	scriptstyle	февра	zaj	kennis	dévelop	кро	־	Ď	ḷ	klass	typen	hund	peint
Layer 19: Mär	Fuß	ٹ	ई	февра	Ζ	VENDOR	peint	biologie	ниш	För	кро	vess	Zent	dévelop	netdev	há	̂	Hö	Frauen	KERN	Ď	scriptstyle	Musik	▸	animated	־	klik	бри	fün	tö	giornata
Layer 20: Mär	Fuß	KERN	ниш	atmos	Ζ	贵	klik	annual	hund	кла	latest	VENDOR	Â	animated	зя	atr	För	vess	急	inaug	Partido	ٹ	peint	февра	Frauen	klass	zu	biologie	tö	dirig	fün
Layer 21: Mär	Fuß	vess	ниш	KERN	fün	biologie	Musik	Ζ	netdev	klik	кро	peint	éc	werken	февра	Bür	atr	ई	ỹ	Partido	VENDOR	newest	Â	För	weit	Frauen	Украї	atmos	ٹ	贵	animated
Layer 22: Fuß	высо	biologie	Ζ	Mär	KERN	ниш	fün	льта	newest	hö	Bür	高	annual	haut	alt	vess	klik	peint	netdev	ỹ	latest	кро	height	atr	éc	塔	февра	atmos	inaug	plaat	bild
Layer 23: Fuß	льта	высо	Ζ	biologie	Mär	atmos	sky	klik	hö	netdev	KERN	кро	fün	peint	newest	éc	vess	ниш	height	Bür	ٹ	bird	alt	plaat	dirig	़	winds	Bird	artikel	ľ	scriptstyle
Layer 24: Fuß	Ζ	sky	scriptstyle	льта	biologie	Υ	высо	Mär	ٹ	़	plaat	kennis	bekan	hö	klik	贵	netdev	Ď	dévelop	Ṭ	éc	Bür	ниш	Ρ	fün	KERN	artikel	peint	vess	▸	ई
Layer 25: Fuß	Ζ	sky	Mär	biologie	ниш	atmos	scriptstyle	льта	Ṭ	kennis	ٹ	Υ	artikel	bekan	Ď	netdev	贵	plaat	़	peint	KERN	éc	февра	ষ	־	dévelop	fün	бри	péri	Ρ	weit
Layer 26: Fuß	Ζ	atmos	sky	kennis	biologie	Mär	Υ	Ṭ	льта	bekan	ниш	scriptstyle	netdev	ٹ	plaat	Ď	贵	־	artikel	péri	़	dévelop	peint	февра	Ρ	бри	Ο	Kultur	représent	représ	vess
Layer 27: sky	Fuß	Ζ	atmos	plaat	Mär	льта	Ď	бри	biologie	bekan	贵	Υ	kennis	̇	ٹ	Ṭ	VENDOR	netdev	vess	Bür	scriptstyle	péri	ẓ	Kultur	мпи	急	ʲ	ниш	़	Ρ	weit
Layer 28: sky	atmos	Ζ	elli	Fuß	бри	icons	льта	̇	Mär	biologie	Υ	ire	televis	贵	Ď	scriptstyle	icy	急	clouds	кор	Bür	визи	дри	artikel	plaat	heav	VENDOR	Luft	bekan	Kultur	Musik
Layer 29: sky	elli	atmos	clouds	ire	icy	icons	бри	sky	Ď	3	air	Bür	televis	verd	Ζ	贵	2		heav	Mär	winds	breeze	Sky	third	визи	mountains	œuvre	̇	Luft	....	cloud
Layer 30: sky	icy	3	2	....	clouds	sky	elli	air	icons	Ď	1	Sky	9	atmos	verd		winds	mountains	outh	�	бри	cloud	8	ql	7	icky	4	bree	blue	vast	贵
Layer 31: sky	icy	3	2	1	9	air	sky	4	clouds	8	blue	7	5	elli	Sky	icons	6	mountains	clear	sk	morning	icky	summer	....	sun	ills	night	cloud	heav	winds	warm
Layer 32: sky	icy	2	3	1	9	4	sk	5	8	7	6	blue	air	sky	Sky	summer	clear	elli	clouds	night	morning	icons	mountains	icky	heav	sun	spring	evening	­	“	....
```

## Metrics
```
Metrics for: 'Birds fly high in the... sky'
Layer N: 		BLEU		Rouge		BERT
Layer 8: 		0.0000		0.0000		0.0000
Layer 16: 		0.0000		0.0000		0.7024
Layer 24: 		0.0000		0.0000		0.6370
Layer 32: 		0.0000		1.0000		1.0000
```

## References
The code wis partially inspired by this repo: [https://www.datacamp.com/tutorial/fine-tuning-llama-2](https://github.com/voidism/DoLa/tree/main)
