QT += core gui\
        widgets

TEMPLATE = app

QMAKE_MOC += -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED
QMAKE_CXXFLAGS += -std=c++0x -stdlib=libc++ -mmacosx-version-min=10.7 #-O4
QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.7

INCLUDEPATH += /usr/local/include \
        /usr/local/Cellar/boost/1.54.0/include

LIBS += \
    -stdlib=libc++ \
    -L/usr/local/lib \
    -lboost_system \
    -lboost_filesystem \
    -lboost_timer

SOURCES += \
    main.cpp

HEADERS += \
    rule.h \
    sequitur.h \
    name_generator.h \
    file_reader.h \
    utils.h

OTHER_FILES += \
    data/achuSeq_1.csv \
    data/achuSeq_2.csv \
    data/achuSeq_3.csv \
    data/achuSeq_4.csv \
    data/achuSeq_5.csv \
    data/achuSeq_6.csv \
    data/achuSeq_7.csv \
    data/achuSeq_8.csv \
    data/achuSeq_9.csv \
    data/oldoSeq_1.csv \
    data/oldoSeq_2.csv \
    data/oldoSeq_3.csv \
    data/oldoSeq_4.csv \
    data/oldoSeq_5.csv \
    data/oldoSeq_6.csv \
    data/oldoSeq_7.csv \
    data/oldoSeq_8.csv \
    data/oldoSeq_9.csv \
    data/oldoSeq_10.csv
