#-------------------------------------------------
#
# Project created by QtCreator 2015-10-01T19:39:55
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = se_project_prototype
CONFIG   += console
CONFIG   -= app_bundle

LIBS += `pkg-config opencv --libs`

TEMPLATE = app


SOURCES += main.cpp \
    utils.cpp

RESOURCES += \
    resources.qrc

HEADERS += \
    utils.h
