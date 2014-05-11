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
    file_reader.h \
    scfg.h \
    stochastic_rule.h \
    in_out_proba.h

OTHER_FILES += \
