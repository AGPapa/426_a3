// set the precision of the float values (necessary if using float)
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
precision mediump int;

// define constant parameters
// EPS is for the precision issue (see precept slide)
#define INFINITY 1.0e+12
#define EPS 1.0e-3

// define constants for scene setting 
#define MAX_LIGHTS 10

// define texture types
#define NONE 0
#define CHECKERBOARD 1
#define MYSPECIAL 2

// define material types
#define BASICMATERIAL 1
#define PHONGMATERIAL 2
#define LAMBERTMATERIAL 3

// define reflect types - how to bounce rays
#define NONEREFLECT 1
#define MIRRORREFLECT 2
#define GLASSREFLECT 3

struct Shape {
    int shapeType;
    vec3 v1;
    vec3 v2;
    float rad;
};

struct Material {
    int materialType;
    vec3 color;
    float shininess;
    vec3 specular;

    int materialReflectType;
    float reflectivity; 
    float refractionRatio;
    int special;

};

struct Object {
    Shape shape;
    Material material;
};

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
    float attenuate;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Intersection {
    vec3 position;
    vec3 normal;
};

// uniform
uniform mat4 uMVMatrix;
uniform int frame;        
uniform float height;
uniform float width;
uniform vec3 camera;
uniform int numObjects;
uniform int numLights;
uniform Light lights[MAX_LIGHTS];
uniform vec3 objectNorm;

varying vec2 v_position;

// find then position some distance along a ray
vec3 rayGetOffset( Ray ray, float dist ) {
    return ray.origin + ( dist * ray.direction );
}

// if a newly found intersection is closer than the best found so far, record the new intersection and return true;
// otherwise leave the best as it was and return false.
bool chooseCloserIntersection( float dist, inout float best_dist, inout Intersection intersect, inout Intersection best_intersect ) {
    if ( best_dist <= dist ) return false;
    best_dist = dist;
    best_intersect.position = intersect.position;
    best_intersect.normal   = intersect.normal;
    return true;
}

// put any general convenience functions you want up here
// ----------- STUDENT CODE BEGIN ------------
// ----------- Our reference solution uses 135 lines of code.

float rand(float n){return fract(sin(n) * 43758.5453123);}

float rand(vec2 n) { 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(float p){
    float fl = floor(p);
  float fc = fract(p);
    return mix(rand(fl), rand(fl + 1.0), fc);
}
    
float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
  vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// ----------- STUDENT CODE END ------------




// forward declaration
float rayIntersectScene( Ray ray, out Material out_mat, out Intersection out_intersect );

// Plane
// this function can be used for plane, triangle, and box
float findIntersectionWithPlane( Ray ray, vec3 norm, float dist, out Intersection intersect ) {
    float a   = dot( ray.direction, norm );
    float b   = dot( ray.origin, norm ) - dist;
    
    if ( a < 0.0 && a > 0.0 ) return INFINITY;
    
    float len = -b/a;
    if ( len < EPS ) return INFINITY;

    intersect.position = rayGetOffset( ray, len );
    intersect.normal   = norm;
    return len;
}

// Triangle
float findIntersectionWithTriangle( Ray ray, vec3 t1, vec3 t2, vec3 t3, out Intersection intersect ) {
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 22 lines of code.
	vec3 norm = normalize(cross(t1 - t2, t1 - t3));
	float dist = dot(norm, t1)/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
	float planeDist = findIntersectionWithPlane(ray, norm, dist, intersect);
	
	//for each side
	vec3 p = ray.origin + ray.direction * planeDist;
	vec3 v1 = t1 - p;
	vec3 v2 = t2 - p;
	vec3 n1 = normalize(cross(v2, v1));
	if (dot(ray.direction, n1) < 0.0)
		return INFINITY;
		
	v1 = t2 - p;
	v2 = t3 - p;
	n1 = normalize(cross(v2, v1));
	if (dot(ray.direction, n1) < 0.0)
		return INFINITY;
		
	v1 = t3 - p;
	v2 = t1 - p;
	n1 = normalize(cross(v2, v1));
	if (dot(ray.direction, n1) < 0.0)
		return INFINITY;
		
	return planeDist;
	
	
//    return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------
}

// Sphere
float findIntersectionWithSphere( Ray ray, vec3 center, float radius, out Intersection intersect ) {   

    vec3 L = center - ray.origin;
    float tca = dot( L, ray.direction );
    if (tca < 0.0) {
        return INFINITY;
    }
    float d = sqrt(dot(L, L) - dot(tca, tca));
    if (d < 0.0) {
        return INFINITY;
    }
    float thc = sqrt(radius * radius - d * d);

    float t1 = tca - thc;
    float t2 = tca + thc;

    if (t1 > 0.0) {
        intersect.position = rayGetOffset( ray, t1);
        intersect.normal = normalize(intersect.position - center);
        return t1;
    }
    else if (t2 > 0.0) {
        intersect.position = rayGetOffset( ray, t2);
        intersect.normal = normalize(intersect.position - center);
        return t2;
    } 
    return INFINITY;

    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 23 lines of code.
    //return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------
}

// Box
float findIntersectionWithBox( Ray ray, vec3 pmin, vec3 pmax, out Intersection out_intersect ) {

	return INFINITY;

    vec3 finalNorm = vec3(0, 0, 0);

    //front
    vec3 v1 = vec3(pmin.x, pmin.y, pmin.z);
    vec3 v2 = vec3(pmin.x, pmin.y, pmax.z);
    vec3 v3 = vec3(pmax.x, pmax.y, pmax.z);
    vec3 norm = normalize(cross(v1 - v2, v3 - v2));
    float D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    float s = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    finalNorm = norm;

    //bottom
    v2 = vec3(pmin.x, pmax.y, pmin.z);
    v3 = vec3(pmax.x, pmax.y, pmin.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    float s2 = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    if (s2 < s) {
        s = s2;
        finalNorm = norm;
    }

    //right side
    v1 = vec3(pmax.x, pmin.y, pmin.z);
    v2 = vec3(pmax.x, pmin.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    s2 = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    if (s2 < s) {
        s = s2;
        finalNorm = norm;
    }

    //back
    v1 = vec3(pmin.x, pmax.y, pmin.z);
    v2 = vec3(pmin.x, pmax.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    s2 = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    if (s2 < s) {
        s = s2;
        finalNorm = norm;
    }

    //top
    v1 = vec3(pmin.x, pmin.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    s2 = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    if (s2 < s) {
        s = s2;
        finalNorm = norm;
    }

    //left side
    v1 = vec3(pmin.x, pmin.y, pmin.z);
    v2 = vec3(pmin.x, pmin.y, pmax.z);
    v3 = vec3(pmin.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    D = -norm.x * pmin.x -norm.y * pmin.y - norm.z * pmin.z;
    s2 = -1.0 * (( D + dot( norm, ray.origin ) ) / dot( norm, ray.direction ));
    if (s2 < s) {
        s = s2;
        finalNorm = norm;
    }

    /*vec3 v1 = vec3(pmin.x, pmin.y, pmin.z);
    vec3 v2 = vec3(pmin.x, pmin.y, pmax.z);
    vec3 v3 = vec3(pmax.x, pmax.y, pmax.z);
    

    vec3 closestFace[4];

    //front
    vec3 norm = normalize(cross(v1 - v2, v3 - v2));
    float dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    float planeDist = findIntersectionWithPlane(ray, norm, dist, out_intersect); 
    closestFace[0] = v1;
    closestFace[1] = v2;
    closestFace[2] = v3;
    closestFace[3] = vec3(pmax.x, pmin.y, pmin.z);

    //bottom
    v2 = vec3(pmin.x, pmax.y, pmin.z);
    v3 = vec3(pmax.x, pmax.y, pmin.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    float newDist = findIntersectionWithPlane(ray, norm, dist, out_intersect);
    if (newDist < planeDist) {
        planeDist = newDist;
    }

    //right side
    v1 = vec3(pmax.x, pmin.y, pmin.z);
    v2 = vec3(pmax.x, pmin.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    newDist = findIntersectionWithPlane(ray, norm, dist, out_intersect);
    if (newDist < planeDist) {
        planeDist = newDist;
        closestFace[3] = vec3(pmax.x, pmax.y, pmin.z);
    }

    //back
    v1 = vec3(pmin.x, pmax.y, pmin.z);
    v2 = vec3(pmin.x, pmax.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    newDist = findIntersectionWithPlane(ray, norm, dist, out_intersect);
    if (newDist < planeDist) {
        planeDist = newDist;
    }

    //top
    v1 = vec3(pmin.x, pmin.y, pmax.z);
    v3 = vec3(pmax.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    newDist = findIntersectionWithPlane(ray, norm, dist, out_intersect);
    if (newDist < planeDist) {
        planeDist = newDist;
        closestFace[3] = vec3(pmax.x, pmin.y, pmax.z);
    }

    //left side
    v1 = vec3(pmin.x, pmin.y, pmin.z);
    v2 = vec3(pmin.x, pmin.y, pmax.z);
    v3 = vec3(pmin.x, pmax.y, pmax.z);
    norm = normalize(cross(v3 - v2, v1 - v2));
    dist = abs((norm.x * v2.x) + (norm.y * v2.y) + (norm.z * v2.z))/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
    newDist = findIntersectionWithPlane(ray, norm, dist, out_intersect);
    if (newDist < planeDist) {
        planeDist = newDist;
        closestFace[3] = vec3(pmin.x, pmax.y, pmin.z);
    }*/

    
    vec3 p = rayGetOffset(ray, s);

    if (p.x < pmin.x || p.y < pmin.y || p.z < pmin.z) {
        return INFINITY;
    }
    if (p.x > pmax.x || p.y > pmax.y || p.z > pmax.z) {
        return INFINITY;
    }

    out_intersect.position = p;
    out_intersect.normal = finalNorm;

    return s;
    // ----------- STUDENT CODE BEGIN ------------
    // pmin and pmax represent two bounding points of the box
    // pmin stores [xmin, ymin, zmin] and pmax stores [xmax, ymax, zmax]
    // ----------- Our reference solution uses 24 lines of code.
    //return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------


}  

// Cylinder
float getIntersectOpenCylinder( Ray ray, vec3 center, vec3 axis, float len, float rad, out Intersection intersect ) {
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 31 lines of code.
    return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------
}

float getIntersectDisc( Ray ray, vec3 center, vec3 norm, float rad, out Intersection intersect ) {
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 15 lines of code.
	float dist = dot(norm, center)/(sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z));
	float planeDist = findIntersectionWithPlane(ray, norm, dist, intersect);
	
	vec3 p = ray.origin + ray.direction * planeDist;
	
	if (distance(p, center) > rad) return INFINITY;
	else return planeDist;
	
	
    return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------
}


float findIntersectionWithCylinder( Ray ray, vec3 center, vec3 apex, float radius, out Intersection out_intersect ) {
    vec3 axis = apex - center;
    float len = length( axis );
    axis = normalize( axis );

    Intersection intersect;
    float best_dist = INFINITY;
    float dist;

    // -- infinite cylinder
    dist = getIntersectOpenCylinder( ray, center, axis, len, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    // -- two caps
    dist = getIntersectDisc( ray, center, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );
    dist = getIntersectDisc( ray,   apex, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    return best_dist;
}
    
// Cone
float getIntersectOpenCone( Ray ray, vec3 apex, vec3 axis, float len, float radius, out Intersection intersect ) {
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 31 lines of code.
    return INFINITY; // currently reports no intersection
    // ----------- STUDENT CODE END ------------
}

float findIntersectionWithCone( Ray ray, vec3 center, vec3 apex, float radius, out Intersection out_intersect ) {
    vec3 axis   = center - apex;
    float len   = length( axis );
    axis = normalize( axis );
        
    // -- infinite cone
    Intersection intersect;
    float best_dist = INFINITY;
    float dist;

    // -- infinite cone
    dist = getIntersectOpenCone( ray, apex, axis, len, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    // -- caps
    dist = getIntersectDisc( ray, center, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    return best_dist;
}

#define MAX_RECURSION 8

vec3 calculateSpecialDiffuseColor( Material mat, vec3 posIntersection, vec3 normalVector ) {
    // ----------- STUDENT CODE BEGIN ------------
    if ( mat.special == CHECKERBOARD ) {
        // do something here for checkerboard
		float x = posIntersection.x;
		float y = posIntersection.y;
		float z = posIntersection.z;
		vec3 color;
		bool black;
		if (mod(x, 16.0) >= 8.0) {
			black = false;
		} else {
			black = true;
		}
		if (mod(y, 16.0) >= 8.0) {
			black = !black;
		}
		if (mod(z, 16.0) >= 8.0) {
			black = !black;
		}
		if (black) {
			color = mat.color*0.3;
		} else {
			color = mat.color*0.8;
		}
		return color;
        // ----------- Our reference solution uses 21 lines of code.
    } 
    else if ( mat.special == MYSPECIAL ) {
        // do something here for myspecial

        vec2 vec = vec2(posIntersection.x, posIntersection.y);
        return vec3(noise(vec));
        // ----------- Our reference solution uses 2 lines of code.
    }

    return mat.color; // special materials not implemented. just return material color.
    // ----------- STUDENT CODE END ------------
}

vec3 calculateDiffuseColor( Material mat, vec3 posIntersection, vec3 normalVector ) {
    // Special colors
    if ( mat.special != NONE ) {
        return calculateSpecialDiffuseColor( mat, posIntersection, normalVector ); 
    }
    return vec3( mat.color );
}

// check if position pos in in shadow with respect to a particular light.
// lightVec is the vector from that position to that light
bool pointInShadow( vec3 pos, vec3 lightVec ) {

//	return false;

    Material material;
    Intersection intersect;
    Ray light = Ray(pos, normalize(lightVec));

    float distToLight = rayIntersectScene(light, material, intersect);
    
    if (distToLight == INFINITY) return false;
	if (length(lightVec) < distToLight) return false;
	else return true;
	
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 10 lines of code.
    // ----------- STUDENT CODE END ------------
}

vec3 getLightContribution( Light light, Material mat, vec3 posIntersection, vec3 normalVector, vec3 eyeVector, bool phongOnly, vec3 diffuseColor ) {

    vec3 lightVector = light.position - posIntersection;
    
    if ( pointInShadow( posIntersection, lightVector ) ) {
        return vec3( 0.0, 0.0, 0.0 );
    }

    if ( mat.materialType == PHONGMATERIAL || mat.materialType == LAMBERTMATERIAL ) {
        vec3 contribution = vec3( 0.0, 0.0, 0.0 );

        // get light attenuation
        float dist = length( lightVector );
        float attenuation = light.attenuate * dist * dist;

        float diffuseIntensity = max( 0.0, dot( normalVector, lightVector ) ) * light.intensity;
        
        // glass and mirror objects have specular highlights but no diffuse lighting
        if ( !phongOnly ) {
            contribution += diffuseColor * diffuseIntensity * light.color / attenuation;
        }
        
        if ( mat.materialType == PHONGMATERIAL ) {
            // ----------- STUDENT CODE BEGIN ------------
            vec3 phongTerm = vec3( 0.0, 0.0, 0.0 ); // not implemented yet, so just add black   
            // ----------- Our reference solution uses 10 lines of code.
            // ----------- STUDENT CODE END ------------
            contribution += phongTerm;
        }

        return contribution;
    }
    else {
        return diffuseColor;
    }

}

vec3 calculateColor( Material mat, vec3 posIntersection, vec3 normalVector, vec3 eyeVector, bool phongOnly ) {
	vec3 diffuseColor = calculateDiffuseColor( mat, posIntersection, normalVector );

	vec3 outputColor = vec3( 0.0, 0.0, 0.0 ); // color defaults to black when there are no lights
	
    for ( int i=0; i<MAX_LIGHTS; i++ ) {

        if( i>=numLights ) break; // because GLSL will not allow looping to numLights
		
        outputColor += getLightContribution( lights[i], mat, posIntersection, normalVector, eyeVector, phongOnly, diffuseColor );
	}
	
	return outputColor;
}

// find reflection or refraction direction ( depending on material type )
vec3 calcReflectionVector( Material material, vec3 direction, vec3 normalVector, bool isInsideObj ) {
    if( material.materialReflectType == MIRRORREFLECT ) {
        return reflect( direction, normalVector );
    }
    // the material is not mirror, so it's glass.
    // compute the refraction direction...
    
    // ----------- STUDENT CODE BEGIN ------------
    // see lecture 13 slide ( lighting ) on Snell's law
    // the eta below is eta_i/eta_r
    float eta = ( isInsideObj ) ? 1.0/material.refractionRatio : material.refractionRatio;
    // ----------- Our reference solution uses 11 lines of code.
    
    return reflect( direction, normalVector ); // return mirror direction so you can see something
    // ----------- STUDENT CODE END ------------
}

vec3 traceRay( Ray ray ) {
    Material hitMaterial;
    Intersection intersect;

    vec3 resColor  = vec3( 0.0, 0.0, 0.0 );
    vec3 resWeight = vec3( 1.0, 1.0, 1.0 );
    
    bool isInsideObj = false;

    for ( int depth = 0; depth < MAX_RECURSION; depth++ ) {
        
        float hit_length = rayIntersectScene( ray, hitMaterial, intersect );
            
        if ( hit_length < EPS || hit_length >= INFINITY ) break;

        vec3 posIntersection = intersect.position;
        vec3 normalVector    = intersect.normal;

        vec3 eyeVector = normalize( ray.origin - posIntersection );           
        if ( dot( eyeVector, normalVector ) < 0.0 )
            { normalVector = -normalVector; isInsideObj = true; }
        else isInsideObj = false;

        bool reflective = ( hitMaterial.materialReflectType == MIRRORREFLECT || 
                            hitMaterial.materialReflectType == GLASSREFLECT );
		vec3 outputColor = calculateColor( hitMaterial, posIntersection, normalVector, eyeVector, reflective );

        float reflectivity = hitMaterial.reflectivity;

        // check to see if material is reflective ( or refractive )
        if ( !reflective || reflectivity < EPS ) {
            resColor += resWeight * outputColor;
            break;
        }
        
        // bounce the ray
        vec3 reflectionVector = calcReflectionVector( hitMaterial, ray.direction, normalVector, isInsideObj );
        ray.origin = posIntersection;
        ray.direction = normalize( reflectionVector );

        // add in the color of the bounced ray
        resColor += resWeight * outputColor;
        resWeight *= reflectivity;
    }

    return resColor;
}

void main( ) {
    float cameraFOV = 0.8;
    vec3 direction = vec3( v_position.x * cameraFOV * width/height, v_position.y * cameraFOV, 1.0 );

    Ray ray;
	ray.origin    = vec3( uMVMatrix * vec4( camera, 1.0 ) );
    ray.direction = normalize( vec3( uMVMatrix * vec4( direction, 0.0 ) ) );

    // trace the ray for this pixel
    vec3 res = traceRay( ray );
    
    // paint the resulting color into this pixel
    gl_FragColor = vec4( res.x, res.y, res.z, 1.0 );
}

