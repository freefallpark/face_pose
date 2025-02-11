// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
 * @file poseTypeObjectSupport.cxx
 * Source file containing the implementation to register the TypeObject representation of the described types in the IDL file
 *
 * This file was generated by the tool fastddsgen.
 */

#include "poseTypeObjectSupport.hpp"

#include <mutex>
#include <string>

#include <fastcdr/xcdr/external.hpp>
#include <fastcdr/xcdr/optional.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/log/Log.hpp>
#include <fastdds/dds/xtypes/common.hpp>
#include <fastdds/dds/xtypes/type_representation/ITypeObjectRegistry.hpp>
#include <fastdds/dds/xtypes/type_representation/TypeObject.hpp>
#include <fastdds/dds/xtypes/type_representation/TypeObjectUtils.hpp>

#include "pose.hpp"


using namespace eprosima::fastdds::dds::xtypes;

namespace re {
namespace face_pose {
// TypeIdentifier is returned by reference: dependent structures/unions are registered in this same method
void register_Pose_type_identifier(
        TypeIdentifierPair& type_ids_Pose)
{

    ReturnCode_t return_code_Pose {eprosima::fastdds::dds::RETCODE_OK};
    return_code_Pose =
        eprosima::fastdds::dds::DomainParticipantFactory::get_instance()->type_object_registry().get_type_identifiers(
        "re::face_pose::Pose", type_ids_Pose);
    if (eprosima::fastdds::dds::RETCODE_OK != return_code_Pose)
    {
        StructTypeFlag struct_flags_Pose = TypeObjectUtils::build_struct_type_flag(eprosima::fastdds::dds::xtypes::ExtensibilityKind::APPENDABLE,
                false, false);
        QualifiedTypeName type_name_Pose = "re::face_pose::Pose";
        eprosima::fastcdr::optional<AppliedBuiltinTypeAnnotations> type_ann_builtin_Pose;
        eprosima::fastcdr::optional<AppliedAnnotationSeq> ann_custom_Pose;
        CompleteTypeDetail detail_Pose = TypeObjectUtils::build_complete_type_detail(type_ann_builtin_Pose, ann_custom_Pose, type_name_Pose.to_string());
        CompleteStructHeader header_Pose;
        header_Pose = TypeObjectUtils::build_complete_struct_header(TypeIdentifier(), detail_Pose);
        CompleteStructMemberSeq member_seq_Pose;
        CompleteStructType struct_type_Pose = TypeObjectUtils::build_complete_struct_type(struct_flags_Pose, header_Pose, member_seq_Pose);
        if (eprosima::fastdds::dds::RETCODE_BAD_PARAMETER ==
                TypeObjectUtils::build_and_register_struct_type_object(struct_type_Pose, type_name_Pose.to_string(), type_ids_Pose))
        {
            EPROSIMA_LOG_ERROR(XTYPES_TYPE_REPRESENTATION,
                    "re::face_pose::Pose already registered in TypeObjectRegistry for a different type.");
        }
    }
}

} // namespace face_pose

} // namespace re

