QT += core gui\
        widgets

TEMPLATE = app

QMAKE_MOC += -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED
QMAKE_CXXFLAGS += -std=c++0x #-stdlib=libc++ -mmacosx-version-min=10.7 #-O4

INCLUDEPATH += /usr/local/lib
        #/usr/local/include \
        #/usr/local/Cellar/boost/1.54.0/include

LIBS += \
    #-stdlib=libc++ \
    #-L/usr/local/lib \
    -L/usr/lib \
    -lboost_system \
    -lboost_filesystem

SOURCES += \
    main.cpp

HEADERS += file_reader.h \
    string_utils.h \
    reduce_utils.h \
    divergence_metrics.h \
    preprocessing.h \
    decision_making.h \
    proba_sequitur.h \
    print_tools.h \
    data.h

OTHER_FILES += \
