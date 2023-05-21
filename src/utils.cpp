#include "utils.h"


std::vector<BBox> utils::sorted_box(std::vector<BBox> bboxes){
    
    int num_boxes = bboxes.size();
    std::sort( bboxes.begin( ), bboxes.end( ), [ ]( const BBox& bbox1, const BBox& bbox2 )
    {
        return bbox1.top_left.y < bbox2.top_left.y;
    });

    // for (auto b : bboxes) print b<<"\n";

    // for i in range(num_boxes - 1):
    //     for j in range(i, 0, -1):
    //         if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and 
    //                 (_boxes[j + 1][0][0] < _boxes[j][0][0]):
    //             tmp = _boxes[j]
    //             _boxes[j] = _boxes[j + 1]
    //             _boxes[j + 1] = tmp
    //         else:
    //             break
    // return _boxes

    return bboxes;
}