
#include "consistent_mapping/SettingsSemantic.h"

using namespace std;


namespace voxblox {

template<>
float SemanticMappingSettings::readParameter<float>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0.0f;
        }
    }
    else if(!node.isReal()){
        std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.real();
    }
}

template<>
int SemanticMappingSettings::readParameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0;
        }
    }
    else if(!node.isInt()){
        std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.operator int();
    }
}

template<>
string SemanticMappingSettings::readParameter<string>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return string();
        }
    }
    else if(!node.isString()){
        std::cerr << name << " parameter must be a string, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.string();
    }
}

template<>
bool SemanticMappingSettings::readParameter<bool>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return false;
        }
    }
    else if(!node.isString()){
        std::cerr << name << " parameter must be a string, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return (    node.string()=="true" || 
                    node.string()=="True" || 
                    node.string()=="TRUE" );
    }
}

SemanticMappingSettings::SemanticMappingSettings(const std::string &configFile)
{
    //Open settings file
    cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
        cerr << "[ERROR]: could not open configuration file at: " << configFile << endl;
        cerr << "Aborting..." << endl;

        exit(-1);
    }
    else{
        cout << "Loading settings from " << configFile << endl;
    }

    std::string item_name = "";
    bool found = true;

    // read in semantic-instance mapping options 
    item_name = "SemanticMapping.task";
    task_ = readParameter<std::string>(fSettings, item_name, found);
    item_name = "SemanticMapping.use_geo_confidence";
    use_geo_confidence_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.use_label_confidence";
    use_label_confidence_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.inst_association";
    inst_association_ = readParameter<int>(fSettings, item_name, found);
    item_name = "SemanticMapping.data_association";
    data_association_ = readParameter<int>(fSettings, item_name, found);
    item_name = "SemanticMapping.enable_semantic_instance_segmentation";
    enable_semantic_instance_segmentation_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.use_label_propagation";
    use_label_propagation_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.debug";
    debug_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.seg_graph_confidence";
    seg_graph_confidence_ = readParameter<int>(fSettings, item_name, found);
    item_name = "SemanticMapping.use_inst_label_connect";
    use_inst_label_connect_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "SemanticMapping.connection_ratio_th";
    connection_ratio_th_ = readParameter<float>(fSettings, item_name, found);

    // read in geometric mapping options
    item_name = "Mapping.num_threads";
    num_threads_ = readParameter<int>(fSettings, item_name, found);
    if(num_threads_ < 0){ num_threads_ = 1;}
    item_name = "Mapping.voxel_size";
    voxel_size_ = readParameter<float>(fSettings, item_name, found);
    item_name = "Mapping.voxels_per_side";
    voxels_per_side_ = readParameter<int>(fSettings, item_name, found);
    item_name = "Mapping.voxel_carving_enabled";
    voxel_carving_enabled_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "Mapping.allow_clear";
    allow_clear_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "Mapping.truncation_distance_factor";
    truncation_distance_factor_ = readParameter<float>(fSettings, item_name, found);
    item_name = "Mapping.max_ray_length_m";
    max_ray_length_m_ = readParameter<float>(fSettings, item_name, found);
    item_name = "Mapping.min_ray_length_m";
    min_ray_length_m_ = readParameter<float>(fSettings, item_name, found);
    item_name = "Mapping.enable_anti_grazing";
    enable_anti_grazing_ = readParameter<bool>(fSettings, item_name, found);
    item_name = "Mapping.merging_min_overlap_ratio";
    merging_min_overlap_ratio_ = readParameter<float>(fSettings, item_name, found);
    item_name = "Mapping.merging_min_frame_count";
    merging_min_frame_count_ = readParameter<int>(fSettings, item_name, found);
    item_name = "Mapping.enable_semantic_instance_segmentation";
    enable_semantic_instance_segmentation_ = readParameter<bool>(fSettings, item_name, found);

    // results
    item_name = "Result.folder";
    result_folder_ = readParameter<std::string>(fSettings, item_name, found);
    item_name = "Result.save_meshes";
    save_meshes_ = readParameter<bool>(fSettings, item_name, found);
}



}