#version 410
// Phone ISP tail: chroma NR, local tone, sharpening halo, CCM, encode. Downscales from sim res.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uIn;
uniform vec2 uSimPx;
uniform float uSharp, uChromaLod, uShadow, uFrame;
vec3 tap(vec2 p){ return texture(uIn,p).rgb; }
vec3 tonemap(vec3 x){
  x=max(x,0.);
  // local-tone-ish: lift shadows, roll highlights like a phone HDR pipeline
  float l=max(luma(x),1e-5);
  float nl=l*(1.+l/9.)/(1.+l*.72);
  x*=nl/l;
  x=mix(x,vec3(nl),smoothstep(.65,1.7,l)*.65);
  vec3 s=lin2srgb(min(x,1.));
  s=mix(s,s*s*(3.-2.*s),.28);
  return s;
}
void main(){
  vec2 e=uSimPx*.75;
  vec3 c=(tap(uv+vec2(e.x,e.y))+tap(uv+vec2(-e.x,e.y))+tap(uv+vec2(e.x,-e.y))+tap(uv+vec2(-e.x,-e.y)))*.25;
  vec3 blur=textureLod(uIn,uv,uChromaLod).rgb;
  float lc=luma(c), lb=max(luma(blur),1e-5);
  // chroma comes from a blurrier plane (noise becomes soft blotches, as on real phones)
  vec3 chroma=blur/lb;
  vec3 x=mix(c,chroma*lc,.82);
  // USM halo
  vec3 wide=textureLod(uIn,uv,2.2).rgb;
  float lw=luma(wide);
  float det=clamp((lc-lw),-.25,.25);
  x+=(x/max(lc,1e-4))*det*uSharp;
  // CCM: OLED primaries through a phone colour filter array
  x=mat3(1.10,-.02,-.02, -.07,1.05,-.05, -.03,-.03,1.07)*x;
  vec3 s=tonemap(x*uShadow);
  // dither to fight 8-bit banding at the very end
  s+=(hash21(gl_FragCoord.xy+uFrame)-.5)/255.;
  o=vec4(s,1.);
}
