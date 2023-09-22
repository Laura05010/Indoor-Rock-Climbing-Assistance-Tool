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
    
    var centerPosition: SIMD3<Float>{
        [(fromJoint.position.x + toJoint.position.x)//2 , ]
    }
}
