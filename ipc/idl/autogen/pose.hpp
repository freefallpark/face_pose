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
 * @file pose.hpp
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool fastddsgen.
 */

#ifndef FAST_DDS_GENERATED__RE_FACE_POSE_POSE_HPP
#define FAST_DDS_GENERATED__RE_FACE_POSE_POSE_HPP

#include <cstdint>
#include <utility>

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#define eProsima_user_DllExport __declspec( dllexport )
#else
#define eProsima_user_DllExport
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define eProsima_user_DllExport
#endif  // _WIN32

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#if defined(POSE_SOURCE)
#define POSE_DllAPI __declspec( dllexport )
#else
#define POSE_DllAPI __declspec( dllimport )
#endif // POSE_SOURCE
#else
#define POSE_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define POSE_DllAPI
#endif // _WIN32

namespace re {

namespace face_pose {

/*!
 * @brief This class represents the structure Pose defined by the user in the IDL file.
 * @ingroup pose
 */
class Pose
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport Pose()
    {
    }

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~Pose()
    {
    }

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object Pose that will be copied.
     */
    eProsima_user_DllExport Pose(
            const Pose& x)
    {
        static_cast<void>(x);
    }

    /*!
     * @brief Move constructor.
     * @param x Reference to the object Pose that will be copied.
     */
    eProsima_user_DllExport Pose(
            Pose&& x) noexcept
    {
        static_cast<void>(x);
    }

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object Pose that will be copied.
     */
    eProsima_user_DllExport Pose& operator =(
            const Pose& x)
    {

        static_cast<void>(x);

        return *this;
    }

    /*!
     * @brief Move assignment.
     * @param x Reference to the object Pose that will be copied.
     */
    eProsima_user_DllExport Pose& operator =(
            Pose&& x) noexcept
    {

        static_cast<void>(x);

        return *this;
    }

    /*!
     * @brief Comparison operator.
     * @param x Pose object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const Pose& x) const
    {
        static_cast<void>(x);
        return true;
    }

    /*!
     * @brief Comparison operator.
     * @param x Pose object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const Pose& x) const
    {
        return !(*this == x);
    }



private:


};

} // namespace face_pose

} // namespace re

#endif // _FAST_DDS_GENERATED_RE_FACE_POSE_POSE_HPP_

