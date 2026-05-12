#include "global_segment_map/meshing/open_sem_color_map.h"

#include "voxblox/core/color.h"

namespace voxblox {

void OpenSemanticColorMap::getColor(
    const OpenSemanticLabel& open_sem_label,
    Color* color
) {
  CHECK_NOTNULL(color);
  std::map<OpenSemanticLabel, Color>::iterator open_sem_color_map_it;
  {
    std::shared_lock<std::shared_timed_mutex> readerLock(color_map_mutex_);
    open_sem_color_map_it = color_map_.find(open_sem_label);
  }

  if (open_sem_color_map_it != color_map_.end()) {
    *color = open_sem_color_map_it->second;
  } 
  else {
    if (color_cnt_ < color_code_.size()) {
      color->r = color_code_.at(color_cnt_)[0];
      color->g = color_code_.at(color_cnt_)[1];
      color->b = color_code_.at(color_cnt_)[2];
      color->a = 255;
    } 
    else {
      *color = randomColor();
    }
    color_cnt_++;

    std::lock_guard<std::shared_timed_mutex> writerLock(color_map_mutex_);
    color_map_.insert(std::pair<OpenSemanticLabel, Color>(open_sem_label, *color));
  }
}

void OpenSemanticColorMap::setColor(
  const OpenSemanticLabel& open_sem_label,
  const Color* color
) {
  CHECK_NOTNULL(color);
  // check if the entry already exists in the map
  std::map<OpenSemanticLabel, Color>::iterator open_sem_color_map_it;
  {
    std::shared_lock<std::shared_timed_mutex> readerLock(color_map_mutex_);
    open_sem_color_map_it = color_map_.find(open_sem_label);
  }

  if (open_sem_color_map_it != color_map_.end()) {
    open_sem_color_map_it->second = *color;
  }
  else {
    std::lock_guard<std::shared_timed_mutex> writerLock(color_map_mutex_);
    color_map_.insert(std::pair<OpenSemanticLabel, Color>(open_sem_label, *color));
  }
}


}  // namespace voxblox
