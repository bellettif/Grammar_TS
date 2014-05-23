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
    main.cpp \
    rule.cpp

HEADERS += \
    file_reader.h \
    rule.h \
    counter.h \
    k_sequitur.h \
    mem_sandwitch.h \
    name_generator.h

OTHER_FILES += \