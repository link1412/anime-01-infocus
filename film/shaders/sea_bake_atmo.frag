#version 410
// Clear-sky panorama. u: azimuth, v: sqrt(elevation), so the horizon gets most of the rows.
in vec2 uv; out vec4 o;
#include "sea_atmo.glsl"
uniform float uAlt;      // km
uniform int uMode;       // 0 sky panorama, 1 sun colour at the sea surface (rgb = irradiance)
uniform float uCold;
void main(){
  vec3 ro=vec3(0.,RE+uAlt,0.);
  if(uMode==1){
    vec3 od=odSun(ro,uSunDir);
    o=vec4(exp(-extinction(od))*uE0*uWB,1.);
    return;
  }
  vec3 rd=panoDir(uv);
  rd=normalize(rd+vec3(0.,.0006,0.));
  float tm=sphere(ro,rd,RT).y;
  vec3 T;
  vec3 L=scatter(ro,rd,tm,T,64);
  float l=dot(L,vec3(.25,.65,.10));
  o=vec4(mix(L,l*vec3(.965,.99,1.05),uCold),1.);
}
