#version 410
// Camera body: world radiance -> EVF panel framebuffer (sRGB, 4:3 panel, 16:9 image, HUD).
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uWorld, uHud;
uniform float uEV, uISO, uFrame, uZebra, uPeak, uRoll, uRS, uOver, uSat, uBars;
uniform vec2 uShift;
uniform vec3 uPeakCol;
uniform float uPassthrough;   // worlds that bring their own tone (ink, words...)
uniform sampler2D uBgDisp;    // the feed's other world, wherever the world has no monitor
uniform float uHasBgDisp;
float wA=1.;

vec3 tone(vec3 x){
  // gentle S-curve in the spirit of a standard picture profile
  x=max(x,0.);
  vec3 y=(x*(2.51*x+.03))/(x*(2.43*x+.59)+.14);
  return clamp(y,0.,1.);
}

vec3 sampleWorld(vec2 im){
  vec2 q=(im-.5)*vec2(16./9.,1.);
  q=rot(uRoll)*q;
  q.x+=uRS*(im.y-.5);
  q=q/vec2(16./9.,1.)+uShift;
  vec4 s=texture(uWorld,.5+q/uOver);
  wA=s.a;
  return s.rgb;
}

void main(){
  vec3 c=vec3(0.);
  float top=uBars, bot=1.-uBars;
  if(uv.y>top&&uv.y<bot){
    vec2 im=vec2(uv.x,(uv.y-top)/(bot-top));
    vec3 w=sampleWorld(im);
    float n=luma(w);
    vec3 e=w*pow(2.,uEV);
    // sensor noise, relative to ISO
    vec2 g=gauss(gl_FragCoord.xy+uFrame*vec2(13.1,7.7));
    float sig=sqrt(max(luma(e),0.)*.0009+.00002)*uISO;
    e+=g.x*sig+vec3(gauss(gl_FragCoord.xy*1.7+uFrame).x,0.,gauss(gl_FragCoord.yx*1.3+uFrame).y)*sig*.55;
    vec3 t=uPassthrough>.5?w:tone(e);
    float l=luma(t);
    t=mix(vec3(l),t,uSat);
    c=uPassthrough>.5?t:lin2srgb(t);
    if(uHasBgDisp>.5) c=mix(texture(uBgDisp,uv).rgb,c,wA);
    // focus peaking: true high-pass of the live image
    if(uPeak>0.){
      vec2 d=1./vec2(1024.,576.);
      float a=luma(sampleWorld(im+vec2(d.x,0.))),b=luma(sampleWorld(im-vec2(d.x,0.)));
      float e1=luma(sampleWorld(im+vec2(0.,d.y))),e2=luma(sampleWorld(im-vec2(0.,d.y)));
      float lap=abs(4.*n-a-b-e1-e2)/(n+.04);
      float m=smoothstep(.55,1.2,lap)*uPeak;
      c=mix(c,uPeakCol,m);
    }
    // zebra: diagonal stripes on what is clipping
    if(uZebra>0.){
      float z=smoothstep(uZebra-.015,uZebra+.015,luma(c));
      float s=step(.5,fract((gl_FragCoord.x+gl_FragCoord.y+uFrame*1.5)/9.));
      c=mix(c,mix(c*.15,vec3(1.),s),z*.9);
    }
  }
  vec4 h=texture(uHud,uv);
  c=mix(c,h.rgb,h.a);
  o=vec4(c,1.);
}
