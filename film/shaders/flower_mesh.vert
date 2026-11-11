#version 410
#include "flower_common.glsl"
in vec3 in_p,in_n,in_t,in_b; in vec2 in_uv; in float in_w;
uniform mat3 uSway; uniform vec3 uPivot; uniform float uFlut;
uniform int uSpace;         // 0 camera, 1 texture space, 2 light tile
uniform mat3 uLm; uniform vec3 uLo; uniform vec2 uLs;
out vec3 vP,vN,vT,vB; out vec2 vUv; out float vZ,vL;
void main(){
  vec3 p=in_p+in_n*in_w*uFlut;
  p=uPivot+uSway*(p-uPivot);
  vP=p; vN=uSway*in_n; vT=uSway*in_t; vB=uSway*in_b; vUv=in_uv;
  vZ=dot(p-uC,uF);
  vL=0.;
  if(uSpace==1){
    gl_Position=vec4(in_uv*2.-1.,0.,1.);
  } else if(uSpace==2){
    vec3 q=uLm*(p-uLo);      // light basis: x,y across, z along the light's travel
    vL=q.z;
    gl_Position=vec4(q.xy/uLs.x,q.z/uLs.y,1.);
  } else {
    gl_Position=project(p);
  }
}
