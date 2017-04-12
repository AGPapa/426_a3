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

//From https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
float rand(float n){return fract(sin(n * 43758.5453123));}

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
	
	//UNCOMMENT THIS LINE TO ADD ANIMATION
	//center = vec3(center.x, center.y + 1.5 * float(frame), center.z);
	
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

    if (t1 - EPS > 0.0) {
        intersect.position = rayGetOffset( ray, t1);
        intersect.normal = normalize(intersect.position - center);
        return t1;
    }
    else if (t2 - EPS > 0.0) {
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

	int closeIndex = -1;
	float closeDist = INFINITY;

    //min x
    vec3 norm = vec3(1, 0, 0);
    float planeDistMinX = findIntersectionWithPlane(ray, norm, pmin.x, out_intersect);
	vec3 p = rayGetOffset(ray, planeDistMinX);
	if (p.z < pmin.z) planeDistMinX = INFINITY;
	if (p.z > pmax.z) planeDistMinX = INFINITY;
	if (p.y < pmin.y) planeDistMinX = INFINITY;
	if (p.y > pmax.y) planeDistMinX = INFINITY;
	if (planeDistMinX < closeDist) {
		closeDist = planeDistMinX;
		closeIndex = 0;
	} 
	
	//max x
    norm = vec3(1, 0, 0);
    float planeDistMaxX = findIntersectionWithPlane(ray, norm, pmax.x, out_intersect);
	p = rayGetOffset(ray, planeDistMaxX);
	if (p.z < pmin.z) planeDistMaxX = INFINITY;
	if (p.z > pmax.z) planeDistMaxX = INFINITY;
	if (p.y < pmin.y) planeDistMaxX = INFINITY;
	if (p.y > pmax.y) planeDistMaxX = INFINITY;
	if (planeDistMaxX < closeDist) {
		closeDist = planeDistMaxX;
		closeIndex = 1;
	}
	
	//min y
    norm = vec3(0, 1, 0);
    float planeDistMinY = findIntersectionWithPlane(ray, norm, pmin.y, out_intersect);
	p = rayGetOffset(ray, planeDistMinY);
	if (p.z < pmin.z) planeDistMinY = INFINITY;
	if (p.z > pmax.z) planeDistMinY = INFINITY;
	if (p.x < pmin.x) planeDistMinY = INFINITY;
	if (p.x > pmax.x) planeDistMinY = INFINITY;
	if (planeDistMinY < closeDist) {
		closeDist = planeDistMinY;
		closeIndex = 2;
	}
	
	//max y
    norm = vec3(0, 1, 0);
    float planeDistMaxY = findIntersectionWithPlane(ray, norm, pmax.y, out_intersect);
	p = rayGetOffset(ray, planeDistMaxY);
	if (p.z < pmin.z) planeDistMaxY = INFINITY;
	if (p.z > pmax.z) planeDistMaxY = INFINITY;
	if (p.x < pmin.x) planeDistMaxY = INFINITY;
	if (p.x > pmax.x) planeDistMaxY = INFINITY;
	if (planeDistMaxY < closeDist) {
		closeDist = planeDistMaxY;
		closeIndex = 3;
	}
	
	//min z
    norm = vec3(0, 0, 1);
    float planeDistMinZ = findIntersectionWithPlane(ray, norm, pmin.z, out_intersect);
	p = rayGetOffset(ray, planeDistMinZ);
	if (p.y < pmin.y) planeDistMinZ = INFINITY;
	if (p.y > pmax.y) planeDistMinZ = INFINITY;
	if (p.x < pmin.x) planeDistMinZ = INFINITY;
	if (p.x > pmax.x) planeDistMinZ = INFINITY;
	if (planeDistMinZ < closeDist) {
		closeDist = planeDistMinZ;
		closeIndex = 4;
	}
	
	//max z
    norm = vec3(0, 0, 1);
    float planeDistMaxZ = findIntersectionWithPlane(ray, norm, pmax.z, out_intersect);
	p = rayGetOffset(ray, planeDistMaxZ);
	if (p.y < pmin.y) planeDistMaxZ = INFINITY;
	if (p.y > pmax.y) planeDistMaxZ = INFINITY;
	if (p.x < pmin.x) planeDistMaxZ = INFINITY;
	if (p.x > pmax.x) planeDistMaxZ = INFINITY;
	if (planeDistMaxZ < closeDist) {
		closeDist = planeDistMaxZ;
		closeIndex = 5;
	}
	
	if (closeIndex == 0) {
		norm = vec3(1, 0, 0);
		planeDistMinX = findIntersectionWithPlane(ray, norm, pmin.x, out_intersect);
		return planeDistMinX;
	} else if (closeIndex == 1) {
		norm = vec3(1, 0, 0);
		planeDistMaxX = findIntersectionWithPlane(ray, norm, pmax.x, out_intersect);
		return planeDistMaxX;
	} else if (closeIndex == 2) {
		norm = vec3(0, 1, 0);
		planeDistMinY = findIntersectionWithPlane(ray, norm, pmin.y, out_intersect);
		return planeDistMinY;
	} else if (closeIndex == 3) {
		norm = vec3(0, 1, 0);
		planeDistMaxY = findIntersectionWithPlane(ray, norm, pmax.y, out_intersect);
		return planeDistMaxY;
	} else if (closeIndex == 4) {
		norm = vec3(0, 0, 1);
		planeDistMinZ = findIntersectionWithPlane(ray, norm, pmin.z, out_intersect);
		return planeDistMinZ;
	} else if (closeIndex == 5) {
		norm = vec3(0, 0, 1);
		planeDistMaxY = findIntersectionWithPlane(ray, norm, pmax.z, out_intersect);
		return planeDistMaxZ;
	} else 
		return INFINITY;

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
	
	vec3 va = axis;
	vec3 pa = center;
	vec3 p = ray.origin;
	vec3 v = ray.direction;
	vec3 dp = p - pa;
	vec3 p2 = center + axis*len;
	
	float A = length(v-dot(v, va)*va)*length(v-dot(v, va)*va);
	float B = 2.0 * dot(v - dot(v,va)*va, dp - dot(dp, va)*va);
	float C = length(dp - dot(dp, va)*va)*length(dp - dot(dp, va)*va) - rad*rad;
	
	//(-b+-sqrt(b^2-4ac))/(2a)
	
	float i = B*B - 4.0*A*C;
	if (i < 0.0) return INFINITY;

	
	float t1 = (-B + sqrt(i))/(2.0*A);
	float t2 = (-B - sqrt(i))/(2.0*A);
	
	if (t2 - EPS > 0.0 && t2 < t1) {
		vec3 q = rayGetOffset( ray, t2);
		if (dot(va, q - pa) <= 0.0) return INFINITY;
		if (dot(va, q - p2) >= 0.0) return INFINITY;
        intersect.position = q;
        intersect.normal =  normalize(q - (dot(q-center,axis)*axis+center));
        return t2;
    } else if (t1 - EPS > 0.0) {
		vec3 q = rayGetOffset( ray, t1);
		if (dot(va, q - pa) <= 0.0) return INFINITY;
		if (dot(va, q - p2) >= 0.0) return INFINITY;
        intersect.position = q;
	    intersect.normal = normalize(q - (dot(q-center,axis)*axis+center));
        return t1;
    }
	
    return INFINITY;
	
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
		
	vec3 va = axis;
	vec3 pa = apex;
	vec3 p = ray.origin;
	vec3 v = ray.direction;
	vec3 dp = p - pa;
	vec3 p2 = apex + axis*len;
	float a = atan(radius/len);
	
	float A = cos(a)*cos(a)*(length(v-dot(v, va)*va)*length(v-dot(v, va)*va)) - sin(a)*sin(a)*dot(v,va)*dot(v,va);
	float B = (2.0 * cos(a)*cos(a)*dot(v - dot(v,va)*va, dp - dot(dp, va)*va)) - (2.0*sin(a)*sin(a)*dot(v,va)*dot(dp,va));
	float C = cos(a)*cos(a)*length(dp - dot(dp, va)*va)*length(dp - dot(dp, va)*va) - sin(a)*sin(a)*dot(dp,va)*dot(dp,va);
	
	//(-b+-sqrt(b^2-4ac))/(2a)
	
	float i = B*B - 4.0*A*C;
	if (i < 0.0) return INFINITY;

	
	float t1 = (-B + sqrt(i))/(2.0*A);
	float t2 = (-B - sqrt(i))/(2.0*A);
	
	if (t2 - EPS > 0.0 && t2 < t1) {
		vec3 q = rayGetOffset( ray, t2);
		if (dot(va, q - pa) <= 0.0) return INFINITY;
		if (dot(va, q - p2) >= 0.0) return INFINITY;
        intersect.position = q;
		vec3 e = q - pa;
        intersect.normal =  normalize(e - length(e)/(cos(a))*axis);
        return t2;
    } else if (t1 - EPS > 0.0) {
		vec3 q = rayGetOffset( ray, t1);
		if (dot(va, q - pa) <= 0.0) return INFINITY;
		if (dot(va, q - p2) >= 0.0) return INFINITY;
        intersect.position = q;
		vec3 e = q - pa;
	    intersect.normal = normalize(e - length(e)/(cos(a))*axis);
        return t1;
    }

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

        vec2 vec = vec2(posIntersection.x*1.7, posIntersection.y*1.7);
        return vec3(0,noise(vec)*0.3,0);
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

float pointShadowRatio ( vec3 pos, vec3 lightVec ) {
    float count = 0.0;
    int k = 7;
    for (int i = 0; i < 49; i++) {
		float x1 = clamp(rand(vec2(float(i)/float(k),0.2)),0.0,1.0)-0.5;
		float x2 = clamp(rand(vec2(0.2,float(i)/float(k))),0.0,1.0)-0.5;
		float x = 2.0*x1*sqrt(1.0-x1*x1-x2*x2);
		float y = 2.0*x2*sqrt(1.0-x1*x1-x2*x2);
		float z = 1.0-2.0*(x1*x1+x2*x2);
        if (!pointInShadow(pos, lightVec+vec3(x,y,z))) count += 1.0;
    }
    return count / float(k * k);
}

vec3 getLightContribution( Light light, Material mat, vec3 posIntersection, vec3 normalVector, vec3 eyeVector, bool phongOnly, vec3 diffuseColor ) {

    vec3 lightVector = light.position - posIntersection;
    
/*    if ( pointInShadow( posIntersection, lightVector ) ) {
        return vec3( 0.0, 0.0, 0.0 );
    }*/

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
			vec3 L = normalize(lightVector);
			vec3 N = normalize(normalVector);
            vec3 R = normalize(2.0 * dot(L, N)*N - L);
			float v = pow(clamp(dot(R, eyeVector),0.0,1.0),mat.shininess);
			vec3 phongTerm = mat.specular * light.color * v;
			// ----------- Our reference solution uses 10 lines of code.
            // ----------- STUDENT CODE END ------------
            contribution += phongTerm;
        }

        return contribution * pointShadowRatio( posIntersection, lightVector );
    }
    else {
        return diffuseColor * pointShadowRatio( posIntersection, lightVector );
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
    
    float iTheta = acos((dot(direction * -1.0, normalVector))/(length(direction) * length(normalVector)));
    float rTheta = asin(eta * sin(iTheta));
    vec3 T = ((eta * cos(iTheta) - cos(rTheta)) * normalVector) - (eta * (direction * -1.0));

    return T;
    //return reflect( direction, normalVector ); // return mirror direction so you can see something
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

