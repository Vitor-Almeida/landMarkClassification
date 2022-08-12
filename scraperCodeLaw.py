from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import re
import pandas as pd

##tirar nome de lei e nome de pessoas?
## tem ums paradas de [xxxxxxx] tirar tudo isso?
##apagar : .' . 1 .1 * * * . . .

options = Options()
options.add_argument("--headless")
driver = Firefox(executable_path="geckodriver", options = options)
driver.get("https://www.law.cornell.edu/supct/cases/topic.htm")

xpathSubject = "/html/body/main/div/div[{}]/ul/ul[{}]/li[{}]/a"
xpathCase = "/html/body/main/div/p[{}]/a"
xpathTextBody = "/html/body/main/div/div[1]/div/div/div[2]"

subjectLISTONA = [["Abortion","Affirmative Action","Aliens","Armed Services","Attainder","Attorneys","Bankruptcy","Bill of Rights","Birth Control","Borders","Capital Punishment","Censorship","Children","Choice of Law","Citizenship","Civil Rights","Commander in Chief","Commerce Clause","Commercial Speech","Communism","Confessions","Conflict of Laws","Congress","Contract Clause","Courts","Criminal Law","Criminal Procedure","Cruel and Unusual Punishment","Damages","Discrimination","Discrimination Based on Nationality","Double Jeopardy","Due Process","Education","Eighth Amendment","Elections","Eleventh Amendment","Eminent Domain","Employment","Environment","Equal Protection","Establishment of Religion","Evidence","Executive Power","Executive Privilege","Extradition","Federal Courts","Federalism","Fifth Amendment","Fighting Words","First Amendment","Flag Desecration","Foreign Affairs","Forum","Fourteenth Amendment","Fourth Amendment","Freedom of Assembly","Freedom of Association","Freedom of Religion","Freedom of Speech","Freedom of the Press","Full Faith and Credit","Gender","Government Employment","Habeas Corpus","Handicapped","Housing","Immunity","Implied Powers","Import Tariffs","Incorporation","Indians","Insanity","International Law","International Relations","Internet","Investigations","Involuntary Servitude","Judicial Review","Jurisdiction","Jury","Justiciability","Juveniles","Labor","Legislative Policy","Libel","Marriage","Mental Health","Mental Retardation","Minimum Contacts","Monopoly","National Power","National Security","Necessary and Proper","New Deal","Ninth Amendment","Obscenity","Pardon","Pensions","Pledge of Loyalty","Police Power","Political Questions","Political Speech","Power to Tax and Spend","Precedent","Presidency","Prisons","Privacy","Privileges and Immunities","Property","Race","Racial Discrimination","Reapportionment","Regulation","Removal Power","Reproduction","Res Judicata","Right to a Hearing","Right to Bear Arms","Right to Confront Witnesses","Right to Counsel","Right to Travel","Searches and Seizures","Second Amendment","Sedition","Segregation","Self-Incrimination","Separation of Power","Sex Discrimination","Sexuality","Sixth Amendment","Slavery","Social Security","Standing","State Action","States","Sterilization","Supremacy Clause","Symbolic Speech","Takings Clause","Taxation","Tenth Amendment","Testimony","Thirteenth Amendment","Trial by Jury","Veto","Voting","War Powers","Welfare Benefits","Wiretapping","Witnesses"]]
subjectPorLink = []

LISTA_FINAL = [[]]


subjectList = []
topicList = []
textList = []
#[20,90,90,110]
for divLoop in range(20): #como deixar isso aqui dinamico?
    for ulLoop in range(90): #como deixar isso aqui dinamico?
        for liLoop in range(90): #como deixar isso aqui dinamico?
            curXpath = xpathSubject.format(divLoop,ulLoop,liLoop)

            try:
                topicEle = driver.find_element(By.XPATH, curXpath)
                subjectLink = topicEle.get_attribute("href")
            except:
                topicEle = ""
                subjectLink = ""
            
            subjectList.append(subjectLink)

value = ""
while value in subjectList:
    subjectList.remove(value)

print(len(subjectList))

for idx,subject in enumerate(subjectList):

    try:
        driver.get(subject)
    except:
        continue

    for pLoop in range(1,110):
        subjectPorLink.append(subjectLISTONA[0][idx])
        curXpath = xpathCase.format(pLoop)

        try:
            subTopicEle = driver.find_element(By.XPATH, curXpath)
            subjectTopicLink = subTopicEle.get_attribute("href")
        except:
            subTopicEle = ""
            subjectTopicLink = ""

        topicList.append(subjectTopicLink)

cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

for topic in topicList:

    try:
        driver.get(topic)
    except:
        textList.append("")
        continue

    try:
        textEle = driver.find_element(By.XPATH, xpathTextBody)
        textLink = textEle.get_attribute("innerHTML")
    except:
        textEle = ""
        textLink = ""

    textList.append(re.sub(cleanr, '', textLink))

LISTA_FINAL.append(subjectPorLink)
LISTA_FINAL.append(textList)
LISTA_FINAL = [x for x in LISTA_FINAL if x != []]

df = pd.DataFrame(LISTA_FINAL).transpose()
df.columns=['subject', 'text']

df['subject'] = df['subject'].astype(str)
df['text'] = df['text'].astype(str)
df = df[df["text"] != ""]
df['text'] = df['text'].str.replace(r'\n',' ',regex=True)
df.to_csv("scrapee.csv")