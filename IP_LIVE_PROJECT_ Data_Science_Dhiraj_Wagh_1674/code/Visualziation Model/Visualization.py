import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import sys
df = pd.read_csv(r"C:\Users\Admin-pc\Downloads\ip\liveproj\DS_DATESET.csv")
df=df.drop(["Email Address"],axis=1)
df=df.drop(["First Name"],axis=1)
df=df.drop(["Last Name"],axis=1)
df=df.drop(["Zip Code"],axis=1)
df=df.drop(["Contact Number"],axis=1)
df=df.drop(["Emergency Contact Number"],axis=1)
df=df.drop(["University Name"],axis=1)
df=df.drop(["Degree"],axis=1)
df=df.drop(["Course Type"],axis=1)
df=df.drop(["Current Employment Status"],axis=1)
df=df.drop(["Expected Graduation-year"],axis=1)
df=df.drop(["State"],axis=1)
df=df.drop(["Certifications/Achievement/ Research papers"],axis=1)
df=df.drop(["DOB [DD/MM/YYYY]"],axis=1)
df=df.drop(["Link to updated Resume (Google/ One Drive link preferred)"],axis=1)
df=df.drop(["link to Linkedin profile"],axis=1)

lbl=df["Label"].tolist()
x=[]
for i in range(0,len(lbl)):
    if lbl[i]=='eligible':
        x.append(1)
    else:
        x.append(0)

Label=pd.DataFrame({"Label":x})
df=df.drop(["Label"],axis=1)
df=pd.concat([df,Label],axis=1,sort=False)

import matplotlib as mpl
import matplotlib.pyplot as plt
with PdfPages('Visualization-output.pdf') as pdf:


    #1
    dftech=df["Areas of interest"]
    dftechplot=dftech.value_counts().to_frame()
    dftechplot.plot(kind='bar',color="orange")
    plt.xlabel('Areas of Interest')
    plt.ylabel('Number of Applicants')
    plt.title('Technology and Its Applicants')
    plt.grid(True)
    pdf.savefig()
    plt.close()
    
    #2
    df1=df[["Areas of interest","Programming Language Known other than Java (one major)"]]
    area=df1.groupby(["Areas of interest"])
    area1 = area.get_group('Data Science ')
    interestplot=area1["Programming Language Known other than Java (one major)"].value_counts().to_frame()
    interestplot.plot(kind='bar',color="red")


    plt.xlabel('Programming Language Known other than Java (one major)')
    plt.ylabel('Number of Applicants')
    plt.title('Major Language known by Data Science Applicants')
    plt.grid(True)
    pdf.savefig()
    plt.close()


    #3
    dfhow=df["How Did You Hear About This Internship?"]
    dfhowplot=dfhow.value_counts().to_frame()
    dfhowplot.plot(kind='bar',color="green")


    plt.xlabel("How Did You Hear About This Internship?")
    plt.ylabel('Number of Applicants')
    plt.title('News of internship from different mediums')
    plt.grid(True)
    pdf.savefig()
    plt.close()


    # # 4) Students who are in the fourth year and have a CGPA greater than 8.0.

    df2=df[["Which-year are you studying in?","CGPA/ percentage"]]
    cgpa=df2.groupby(["Which-year are you studying in?"])

    fourthyear=cgpa.get_group('Fourth-year')
    fourthyear1=fourthyear.reset_index(drop=True)

    x=[]
    x=fourthyear1["CGPA/ percentage"]
    greaterthan=[]
    for i in range(0,len(fourthyear1)):
        if x[i] >8.0:
            greaterthan.append("YES")
        elif x[i]<8.0 :
            greaterthan.append("NO")
            

    finalcount=pd.DataFrame({"Greater Than 8.0":greaterthan})
    cgpaplot=finalcount["Greater Than 8.0"].value_counts().to_frame()
    cgpaplot.plot(kind='bar',color="purple")


    plt.xlabel("CGPA")
    plt.ylabel('Number of Applicants in Fourth Year')
    plt.title('CGPA of Fourth Year Students')
    plt.grid(True)
    pdf.savefig()
    plt.close()
   
    len(df[(df['Which-year are you studying in?']=="Fourth-year") & (df['CGPA/ percentage']>8.0)])


    # # 5) Students who applied for Digital Marketing with verbal and written communication score greater than 8.


    dfdigital=df[["Areas of interest","Rate your verbal communication skills [1-10]","Rate your written communication skills [1-10]"]]
    grpdigital=dfdigital.groupby(["Areas of interest"])
    digi=grpdigital.get_group("Digital Marketing ")
    digi1=digi.reset_index(drop=True)

    y=digi1["Rate your verbal communication skills [1-10]"].tolist()
    z=digi1["Rate your written communication skills [1-10]"].tolist()
    grt=[]
    for i in range(0,624):
        if y[i]  > 8:
            if z[i]>8:
                grt.append("YES")
            elif z[i]==8:
                grt.append("NO")
            else :
                grt.append("NO")
        elif y[i]==8:
            grt.append("NO")
            
        else:
            grt.append("NO")      

    finalcount1=pd.DataFrame({"Greater Than 8":grt})
    communicationplot=finalcount1["Greater Than 8"].value_counts().to_frame()
    communicationplot.plot(kind='bar',color="grey")
    plt.xlabel("Verbal Rating")
    plt.ylabel('Number of Applicants in Digital Marketing')
    plt.title('Verbal and Written Communication Skills of Digital Marketing Applicants')
    plt.grid(True)
    pdf.savefig()
    plt.close()



    # # 6a) YEAR WISE CLASSIFICATION 


    dfyear=df["Which-year are you studying in?"]
    dfyearplot=dfyear.value_counts().to_frame()
    dfyearplot.plot(kind='bar',color="black")


    plt.xlabel("Year")
    plt.ylabel('Number of Applicants')
    plt.title('YEAR WISE CLASSIFICATION')
    plt.grid(True)
    pdf.savefig()
    plt.close()


    # # 6b) AREA OF STUDY CLASSIFICATION


    dfstudy=df["Major/Area of Study"]
    dfstudyplot=dfstudy.value_counts().to_frame()
    dfstudyplot
    dfstudyplot.plot(kind='barh',color="cyan")
    plt.ylabel("Major/Area of Study")
    plt.xlabel('Number of Applicants')
    plt.title('AREA OF STUDY  CLASSIFICATION')
    plt.grid(True)
    pdf.savefig()
    plt.close()


    # # 7a) CITY WISE CLASSIFICATION
    dfstay=df["City"]
    dfstayplot=dfstay.value_counts().to_frame()
    dfstayplot.plot(kind='bar',color="pink")
    plt.xlabel("City")
    plt.ylabel('Number of Applicants')
    plt.title('CITY BASED CLASSIFICATION')
    plt.grid(True)
    pdf.savefig()
    plt.close()


    # # 7b) COLLEGE WISE CLASSIFICATION
    dfcollege=df["College name"]
    dfcollegeplot=dfcollege.value_counts().to_frame()
    dfcollegeplot.plot(kind='barh',figsize=(8, 5),color="blue")
    plt.ylabel("College Name")
    plt.xlabel('Number of Applicants')
    plt.title('COLLEGE WISE CLASSIFICATION')
    plt.grid(True)
    pdf.savefig()
    plt.close()




    # # 8) Plot the relationship between CGPA and Target Variable


    plt.scatter(df["CGPA/ percentage"],df["Label"])
    plt.ylabel("Target Variable")
    plt.xlabel('CGPA')
    plt.title("Relationship between CGPA And Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()


    df[["CGPA/ percentage", "Label"]].corr()

    # # 9) Plot the relationship between Area of interest and Target Variable


    dfencode=df
    dfencode["Areas of interest"] = dfencode["Areas of interest"].astype('category')
    dfencode["Areas of interest"] = dfencode["Areas of interest"].cat.codes
    plt.scatter(dfencode["Areas of interest"],dfencode["Label"])
    plt.ylabel("Target Variable")
    plt.xlabel('Area of Interest')
    plt.title("Relationship between Area of Interest and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()
    dfencode[["Areas of interest", "Label"]].corr()

    # # 10) Plot the relationship between Year of study ,Major and Target Variable

    dfencode["Which-year are you studying in?"]=dfencode["Which-year are you studying in?"].astype("category")
    dfencode["Which-year are you studying in?"]=dfencode["Which-year are you studying in?"].cat.codes
    dfencode["Major/Area of Study"]=dfencode["Major/Area of Study"].astype("category")
    dfencode["Major/Area of Study"]=dfencode["Major/Area of Study"].cat.codes
    plt.scatter(dfencode["Major/Area of Study"],dfencode["Label"])
    plt.ylabel("Target Variable")
    plt.xlabel('Major')
    plt.title("Relationship between Major of Study and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()
    dfencode[["Major/Area of Study", "Label"]].corr()

    plt.scatter(dfencode["Which-year are you studying in?"],dfencode["Label"])
    plt.ylabel("Target Variable")
    plt.xlabel('Year of Study')
    plt.title("Relationship between Year of Study and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()
    dfencode[["Which-year are you studying in?", "Label"]].corr()

    print("PDF CREATED")

    











