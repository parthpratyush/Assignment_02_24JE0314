from pathlib import Path
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, PageBreak

ROOT=Path(__file__).resolve().parent
def main():
    with (ROOT/'results.csv').open(newline='', encoding='utf-8') as f: rows=list(csv.DictReader(f))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Section2', parent=styles['Heading1'], textColor=colors.HexColor('#17365D')))
    styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'], fontSize=9, leading=13))
    story=[Spacer(1,2*cm), Paragraph('NUMERICAL STUDY OF THE ORR-SOMMERFELD EQUATION',styles['Title']),
      Paragraph('Spatial stability of a Blasius boundary layer - Compound Matrix Method',styles['Heading2']),
      Paragraph('Student: Pratyush Singh Parmar | Roll No.: 24JE0314 | IIT (ISM) Dhanbad',styles['Body2']),Spacer(1,1*cm),
      Paragraph('1. Governing equation',styles['Section2']),
      Preformatted('(D^2-alpha^2)^2 phi = i Re [(alpha U-omega0)(D^2-alpha^2)phi - alpha U\'\' phi]'),
      Paragraph('2. Compound-matrix IVP',styles['Section2']),
      Preformatted('y1\' = y2; y2\' = y3+y4; y3\' = b1*y2+y5; y4\' = y5;\ny5\' = y6+b1*y4+b2*y1; y6\' = b2*y2'),
      Preformatted('Q=sqrt(alpha^2+i Re(alpha-omega0))\ny1=alpha-Q; y2=Q^2-alpha^2; y3=alpha^3-Q^3\ny4=alpha^2*Q-alpha*Q^2; y5=alpha*Q^3-alpha^3*Q; y6=alpha^3*Q^2-alpha^2*Q^3'),
      Paragraph('The six variables are integrated backward from y_max to the wall. The wall dispersion condition is y1(0)=0.',styles['Body2']),PageBreak(),
      Paragraph('3. Numerical results: Re_delta*=1500, omega0=0.1, y_max=12',styles['Section2'])]
    table=[['Mode','alpha_r','alpha_i','|y1(0)|','Behaviour']]
    for i,r in enumerate(rows,1): table.append([str(i),r['alpha_r'],r['alpha_i'],r['abs_wall_residual'],'Growth' if float(r['alpha_i'])<0 else 'Decay'])
    t=Table(table,colWidths=[1.5*cm,3*cm,3*cm,3.5*cm,2.5*cm]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightblue),('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story += [t,Spacer(1,0.5*cm),Paragraph('alpha_i < 0 indicates spatial growth; alpha_i > 0 indicates spatial decay.',styles['Body2'])]
    for name in ['blasius_base_flow.png','dispersion_contours.png','dispersion_magnitude.png','spatial_growth_indicator.png']:
        p=ROOT/'figures'/name
        if p.exists():
            story += [PageBreak(),Paragraph(name.replace('_',' '),styles['Section2']),Image(str(p),width=15*cm,height=8*cm)]
    doc=SimpleDocTemplate(str(ROOT/'REPORT.pdf'),pagesize=A4,rightMargin=1.8*cm,leftMargin=1.8*cm,topMargin=1.7*cm,bottomMargin=1.7*cm)
    doc.build(story)
    print('Created REPORT.pdf')

if __name__=='__main__': main()
