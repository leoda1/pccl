#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Include common build rules
include ../../../../makefiles/common.mk
include ../../../../makefiles/examples.mk

TARGET = allgather

COMMON_INC = ../../common/include
INCLUDES += -I$(COMMON_INC)

SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

help:
	@echo "PCCL Example: AllGather (usage: ./allgather [bytes_per_rank] [iters])"

.PHONY: all test clean help
