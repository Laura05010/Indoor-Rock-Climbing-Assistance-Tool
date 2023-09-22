//
//  SkeletonBone.swift
//  BodyPoseTracker
//
//  Created by Laura M Madrid on 2023-09-22.
//
// These will represent the Bones of the skeleton; which join the joints!

import Foundation
import RealityKit

struct SkeletonBone{
    var fromJoint: SkeletonJoint
    var toJoint: SkeletonJoint
    
    // calculate the midpoint between the joints
    var centerPosition: SIMD3<Float>{
        [(fromJoint.position.x + toJoint.position.x)/2 , (fromJoint.position.y + toJoint.position.y)/2, (fromJoint.position.z + toJoint.position.z)/2]
    }
    
    //Euclidean distance between both points
    var length: Float {
        simd_distance(fromJoint.position, toJoint.position)
    }
}
