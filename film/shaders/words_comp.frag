#version 410
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uAdd,uMask; uniform float uNight,uHasMask,uTime,uAspect;
void main(){
  vec3 add=texture(uAdd,uv).rgb;
  vec2 v=(uv*2.-1.)*vec2(uAspect,1.);
  vec3 c=add;
  // a faint sky: the only thing that is not a character
  c+=vec3(.006,.006,.012)*(.6+.4*smoothstep(1.2,-1.,v.y));
  if(uHasMask>.5){
    float m=clamp(texture(uMask,uv).r*4.4,0.,1.);
    // paper to night: the same marks go from dark on light to light on dark
    vec3 paper=vec3(.90,.80,.625)*(.96+.05*fbm(v*7.,3));
    vec3 ink=vec3(.035,.032,.03);
    vec3 day=mix(paper,ink,m);
    vec3 nite=c+vec3(1.,.74,.42)*m*.62;
    vec3 d=srgb2lin(day);
    c=mix(d,nite,uNight);
  }
  o=vec4(c,1.);
}
