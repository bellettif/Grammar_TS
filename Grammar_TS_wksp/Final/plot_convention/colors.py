'''
Created on 21 juin 2014

@author: francois
'''

achu_colors = {'achuSeq_1' : '#FF6600',
               'achuSeq_2' : '#FF6633',
               'achuSeq_3' : '#FF6666',
               'achuSeq_4' : '#FF9900',
               'achuSeq_5' : '#FF9933',
               'achuSeq_6' : '#FF9966',
               'achuSeq_7' : '#FFCC00',
               'achuSeq_8' : '#FFCC33',
               'achuSeq_9' : '#FFCC66'}

oldo_colors = {'oldoSeq_1' : '#3300CC',
               'oldoSeq_2' : '#3333CC',
               'oldoSeq_3' : '#3366CC',
               'oldoSeq_4' : '#3399CC',
               'oldoSeq_5' : '#3300FF',
               'oldoSeq_6' : '#3333FF',
               'oldoSeq_7' : '#3366FF',
               'oldoSeq_8' : '#3399FF',
               'oldoSeq_9' : '#33CCFF'}

all_colors = dict(achu_colors.items() + 
                  oldo_colors.items())

oranges = achu_colors.values()
blues = oldo_colors.values()