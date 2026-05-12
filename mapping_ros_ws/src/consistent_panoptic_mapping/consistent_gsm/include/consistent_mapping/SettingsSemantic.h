#ifndef SETTINGS_SEMANTIC_H
#define SETTINGS_SEMANTIC_H

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/core/persistence.hpp>

namespace voxblox {

    class SemanticMappingSettings {
    public:
    /*
        * Delete default constructor
        */
    SemanticMappingSettings() = delete;
    /*
        * Constructor from file
        */
    SemanticMappingSettings(const std::string &configFile);

    // get settings
    // semantic-instance mapping settings
    std::string getTask(){return task_;}
    bool getUseGeoConfidence(){return use_geo_confidence_;}
    bool getUseLabelConfidence(){return use_label_confidence_;}
    int getInstAssociateMode(){return inst_association_;}
    int getDataAssociateMode(){return data_association_;}
    bool getUseLabelPropagation(){return use_label_propagation_;}
    bool getUseDebug(){return debug_;}
    int getSegGraphMode(){return seg_graph_confidence_;}
    bool getUseInstLabelConnect(){return use_inst_label_connect_;}
    float getSegGraphConnectRatioTH(){return connection_ratio_th_;}
    int getMergingMinFrame(){return merging_min_frame_count_;}
    bool getUseSemInstSegmentation(){return enable_semantic_instance_segmentation_;}

    // mapping options
    float getNumThreads(){return num_threads_;}
    float getVoxelSize(){return voxel_size_;}
    int getVoxelBlockLen(){return voxels_per_side_;}
    bool getUseVoxelCarving(){return voxel_carving_enabled_;}
    bool getAllowClear(){return allow_clear_;}
    float getTruncateDistFactor(){return truncation_distance_factor_;}
    float getMaxRayLen(){return max_ray_length_m_;}
    float getMinRayLen(){return min_ray_length_m_;}
    bool getUseAntiGrazing(){return enable_anti_grazing_;}
    float getMergingMinOverlapRation(){return merging_min_overlap_ratio_;}

    // result folder
    std::string getResultFolder(){return result_folder_;}
    bool getSaveMeshes(){return save_meshes_;}

    private:

    // load param items
    template<typename T>
    T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return T();
            }

        }
        else{
            found = true;
            return (T) node;
        }
    }

    // semantic-instance mapping options 
    std::string task_; // semantic task, CoCo or CoCoPano
    bool use_geo_confidence_;
    bool use_label_confidence_;
    int inst_association_;
    int data_association_;
    bool enable_semantic_instance_segmentation_;
    bool use_label_propagation_;
    bool debug_;
    int seg_graph_confidence_;
    bool use_inst_label_connect_;
    float connection_ratio_th_;
    float merging_min_overlap_ratio_ = 0.15;
    int merging_min_frame_count_ = 2;

    // mapping options
    int num_threads_ = 1;
    float voxel_size_ = 0.01;
    int voxels_per_side_ = 8;
    bool voxel_carving_enabled_ = false;
    bool allow_clear_ = true;
    float truncation_distance_factor_ = 5.0;
    float max_ray_length_m_ = 3.0;
    float min_ray_length_m_ = 0.1;
    bool enable_anti_grazing_ = false;

    // result folder
    std::string result_folder_ = "";
    bool save_meshes_ = false;
    };
}

#endif //SETTINGS_SEMANTIC_H