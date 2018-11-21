TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    conj_grad_solve.cpp \
    IO.cpp \

HEADERS += \
    conj_grad_solve.hpp \
    IO.hpp \
