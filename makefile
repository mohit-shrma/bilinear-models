CXX=/usr/bin/g++
RM=rm -f

GKINCHOME=/Users/mohitsharma/dev/gklib/trunk
GKLIBHOME=/Users/mohitsharma/dev/gklib/trunk/build/Darwin-x86_64/

EIGENPATH=/Users/mohitsharma/dev/eigen

#Standard Libraries
STDLIBS=-lm -lpthread

#external libraries
EXT_LIBS=-lGKlib  
EXT_LIBS_DIR=-L$(GKLIBHOME) 

CPPFLAGS=-g -O3 -Wall -std=c++11 -I$(GKINCHOME) -I$(EIGENPATH) 
LDFLAGS=-g
LDLIBS=$(STDLIBS) $(EXT_LIBS_DIR) $(EXT_LIBS)  

SRCS=model.cpp modelBPR.cpp mathUtil.cpp util.cpp  main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: bilinear

bilinear: $(OBJS)
	$(CXX) $(LDFLAGS) -o bilinear $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) tool

include .depend




