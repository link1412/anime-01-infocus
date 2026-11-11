#version 410
// One card of the light atlas: an orthographic look at one group of objects, recording where the surface is.
in vec2 uv;
layout(location=0) out vec4 oP;
layout(location=1) out vec4 oN;
uniform int uAxis;
uniform float uSign,uPlane,uLen,uIsRoom,uGroup;
uniform vec4 uDom;
#include "desk_geo.glsl"
void main(){
  vec2 a=mix(uDom.xy,uDom.zw,uv);
  vec3 ro;
  if(uAxis==0) ro=vec3(uPlane,a.x,a.y);
  else if(uAxis==1) ro=vec3(a.x,uPlane,a.y);
  else ro=vec3(a.x,a.y,uPlane);
  vec3 ax=vec3(0.); ax[uAxis]=uSign;
  if(uIsRoom>.5){ oP=vec4(ro,uGroup+1.); oN=vec4(ax,0.); return; }
  float t=0.;
  for(int i=0;i<160;i++){
    float d=map(ro-ax*t);
    if(d<.00012){
      vec3 p=ro-ax*t;
      oP=vec4(p,uGroup+1.);
      oN=vec4(mapNormal(p,.0003),t);
      return;
    }
    t+=d*.8;
    if(t>uLen) break;
  }
  oP=vec4(0.); oN=vec4(0.);
}
