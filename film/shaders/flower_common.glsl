// the camera: a thin lens. a point goes through the piece of the lens this pass is sampling.
uniform vec3 uC,uF,uR,uU;
uniform vec2 uLens;         // metres, on the lens
uniform float uFd,uTan,uZoom,uAspect;
uniform vec2 uGc,uJit;
const float NEAR=.04,FAR=4.;
vec4 project(vec3 P){
  vec3 q=P-uC;
  float z=dot(q,uF);
  vec2 s=(vec2(dot(q,uR),dot(q,uU))-uLens)/z+uLens*uFd;
  vec2 v=(s/uTan-uGc)*uZoom;
  vec2 ndc=v/vec2(uAspect,1.)+uJit;
  float A=(FAR+NEAR)/(FAR-NEAR),B=-2.*FAR*NEAR/(FAR-NEAR);
  return vec4(ndc*z,A*z+B,z);
}
float depthOf(float z){
  float A=(FAR+NEAR)/(FAR-NEAR),B=-2.*FAR*NEAR/(FAR-NEAR);
  return ((A*z+B)/z)*.5+.5;
}
