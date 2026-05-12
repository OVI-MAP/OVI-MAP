#ifndef SEMANTICS_META_DATA_H_
#define SEMANTICS_META_DATA_H_

#include <iostream>
#include "global_segment_map/common.h"

namespace voxblox {

class MetaSemantics
{
public:
    MetaSemantics(std::string class_set = "Nyu40")
    {
        std::map<SemanticLabel, bool> is_thing_Nyu40 = {
            {0, false},
            {1, false},
            {2, false},
            {3, true},
            {4, true},
            {5, true},
            {6, true},
            {7, true},
            {8, true},
            {9, true},
            {10, true},
            {11, true},
            {12, true},
            {13, false},
            {14, true},
            {15, false},
            {16, true},
            {17, false},
            {18, true},
            {19, false},
            {20, false},
            {21, false},
            {22, false},
            {23, false},
            {24, true},
            {25, true},
            {26, false},
            {27, false},
            {28, true},
            {29, false},
            {30, false},
            {31, true},
            {32, true},
            {33, true},
            {34, true},
            {35, true},
            {36, true},
            {37, true},
            {38, false},
            {39, true},
            {40, false}
        };

        if(class_set == "Nyu40")
        {
            is_thing_ = is_thing_Nyu40;
        }
        else
            is_thing_ = is_thing_Nyu40;
    }

    bool isThing(SemanticLabel semantic_label)
    {
        std::cout << "Getting 'isThing' for sem_label: " << semantic_label << std::endl;
        if(is_thing_.find(semantic_label) != is_thing_.end())
            return is_thing_[semantic_label];
        else
            return false;
    }

    std::map<SemanticLabel, bool> is_thing_;
};
}



#endif //SEMANTICS_META_DATA_H_