#	Karishma Sinha
#	2018339
#	Natural Language Interface for Career Advisory Prolog Program


import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from flair.models import TextClassifier
from flair.data import Sentence
emotion_classifier = TextClassifier.load('en-sentiment')




print("Holland Code Determination\n")
print("......................................")

# print("Realistic\n") 
# print("Investigative\n")
# print("Artistic\n")
# print("Social\n")
# print("Enterprising\n")	
# print("Conventional\n")	
#'investigative'. 	
#'artistic'. 	
#'social'. 	
#'enterprising'. 	
#'conventional'.
inputlist = []
#f = open("HollandQualitiesPrologfacts.txt", 'w')


inp0=input("How would you describe your satisfaction with your current educational qualification?\n")
input_sentence0 = Sentence(inp0)
emotion_classifier.predict(input_sentence0)
print('Sentence above is: ', input_sentence0.labels)
f = open("HollandQualitiesPrologfacts.txt", 'w')
if str(input_sentence0.labels)[1]=="N":
	print("-1")
	f.write("satisfied_w_education(no).\n")
else:
	print("1")
	f.write("satisfied_w_education(yes).\n")


inp01=input("How would you describe your satisfaction with your current area of work\n")
#inp1 = input("How would you describe your qualities and how they would fit in a work place? ")


input_sentence01 = Sentence(inp01)

emotion_classifier.predict(input_sentence01)
print('Sentence above is: ', input_sentence01.labels)
if str(input_sentence01.labels)[1]=="N":
	print("-1")
	f.write("satisfied_w_areaofwork(no).\n")
else:
	print("1")
	f.write("satisfied_w_areaofwork(yes).\n")

inp1 = input("How would you describe your satisfaction with your current educational qualification?\n")

# input_sentence = Sentence(inp1)
# emotion_classifier.predict(input_sentence)


# print sentence with predicted labels
#print('Sentence above is: ', sentence.labels)
# print("\nWe have got ...", inp1)
tok1 = word_tokenize(inp1)
#print("\n\n...tokens are ...", tok1)

ps = PorterStemmer()
for wod in tok1:
    #print("\n..word is..",wod)
    stem1 = ps.stem(wod)
    #print("...stem is ...", stem1)
    inputlist.append(stem1)

print("\n Your entered List of Tokens is: ", inputlist)

# myList = ['one', 'six','ten']
# str = "one two three four five"
# if any(x in str for x in myList):
#     print ("Found a match")
# else:
#     print ("Not a match")
strinterest ="interest like am enjoy good nice love want hope desire work"
strreal="realist realistic practical rational logical technical goals achievable tasks micromanage"
strart="artistic art creative paint draw imagination"
strinvestigative="investigative investigate find know discover unravel analyze"
strsocial="social society mankind greater purpose meaningful people"
strenterprising="competition compete hard enterprising enterprise entrepreneur business profit"
strconventional="convention conventional salary job norms"

countr=0
counta=0
counti=0
counts=0
counte=0
countc=0


for x in inputlist:
	if x in strinterest:
		#print(x)
		for y in inputlist:

			if y in strreal:
				countr+=1
				if countr==1:
					f.write("qual(realistic).\n")
				#print(x,y)
			if y in strart: 
				counta+=1
				if counta==1:
					f.write("qual(artistic).\n")
				#print(x,y)
			if y in strinvestigative: 
				counti+=1
				if counti==1:
					f.write("qual(investigative).\n")
				#print(x,y)
			if y in strsocial:
				counts+=1
				if counts==1:
					f.write("qual(social).\n")
				#print(x,y)
			if y in strenterprising:
				counte+=1
				if counte==1:
					f.write("qual(enterprising).\n")
				#print(x,y)
			if y in strconventional:
				countc+=1
				if countc==1:
					f.write("qual(conventional).\n")
				#print(x,y)
	#break

# if "am" in inplist:
#     if "adventur" in inplist:
#         f.write("interest_in(adventure).")

f.close()
