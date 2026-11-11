#version 410
// Phone camera looking into the eyepiece. Each phone photosite integrates the
// OLED sub-pixels through eyepiece and phone optics -> moire/screen door emerge.
#include "common.glsl"
out vec4 o;
uniform float uNoise,uDustK,uFlat;
uniform sampler2D uPanel, uRows, uDust;
uniform vec2 uRes, uPanelC, uPupil, uTilt;
uniform float uPanelScale, uRoll, uK1, uK2, uCA, uDefocus, uFC, uGain, uFrame, uMask, uPupilR, uPupilSoft;
uniform float uBandDepth, uBandCount, uBandPhase, uVeil, uGhost, uRubber, uPSF;

const int N=10;
const vec2 OFF[N]=vec2[N](vec2(-.42,-.31),vec2(.11,-.46),vec2(.44,-.17),vec2(-.18,-.08),vec2(.27,.09),
  vec2(-.46,.18),vec2(-.06,.29),vec2(.4,.36),vec2(-.29,.45),vec2(.08,-.21));
const vec2 PX=vec2(1024.,768.);

vec2 toPanel(vec2 fragPx,int ch){
  float ry=clamp(fragPx.y/uRes.y,0.,1.);
  vec2 rs=texture(uRows,vec2(.5,ry)).rg;       // hand shake sampled along the rolling readout
  vec2 q=(fragPx-.5*uRes)/uRes.y+rs;
  vec2 a=rot(uRoll)*(q-uPanelC)/uPanelScale;
  a/=1.+dot(a,uTilt);
  float r2=dot(a,a);
  a*=1.+uK1*r2+uK2*r2*r2;
  float ca=ch==0?-uCA:(ch==2?uCA:0.);
  a*=1.+ca*r2;
  return a;
}

void main(){
  ivec2 ip=ivec2(gl_FragCoord.xy);
  int ch=(ip.x&1)==0?((ip.y&1)==0?0:1):((ip.y&1)==0?1:2);
  vec2 a0=toPanel(gl_FragCoord.xy,ch);
  float rad=length(a0);
  float blur=uDefocus+uFC*rad*rad;              // field curvature softens the corners
  float lod=blur>1.1?log2(blur*.85):-1.;
  float acc=0.;
  vec2 uvc=vec2(a0.x/(4./3.),a0.y)+.5;
  for(int k=0;k<N;k++){
    vec2 s=gl_FragCoord.xy+OFF[k]*(uPSF+blur*1.6);
    vec2 a=toPanel(s,ch);
    vec2 u=vec2(a.x/(4./3.),a.y)+.5;
    if(u.x<0.||u.x>1.||u.y<0.||u.y>1.) continue;
    float v;
    if(uFlat>.5){
      v=pow(textureLod(uPanel,u,0.)[ch],2.2);
    } else if(lod>=0.){
      v=pow(textureLod(uPanel,u,lod)[ch],2.2);
    } else {
      vec2 tc=u*PX; ivec2 t=ivec2(floor(tc)); vec2 f=fract(tc);
      float fetched=pow(texelFetch(uPanel,t,0)[ch],2.2);
      float lo=float(ch)/3.+.035, hi=float(ch+1)/3.-.035;
      float m=step(lo,f.x)*step(f.x,hi)*step(.06,f.y)*step(f.y,.94);
      m/=(hi-lo)*.88;
      v=fetched*mix(1.,m,uMask);
    }
    acc+=v;
  }
  float L=acc/float(N);
  // exit pupil / eyepiece aperture clipping seen from an off-axis phone
  float vig=1.-smoothstep(uPupilR-uPupilSoft,uPupilR+uPupilSoft,length(a0-uPupil));
  vig*=1.-.28*smoothstep(.15,.95,rad);
  // eyepiece dust sits close to the panel image plane
  vec2 du=vec2(a0.x/(4./3.),a0.y)+.5;
  L*=mix(1.,texture(uDust,du).r,.85*uDustK);
  L*=vig;
  // refresh beat between OLED scan and rolling shutter
  float ry=gl_FragCoord.y/uRes.y;
  L*=1.+uBandDepth*sin(6.28318*(ry*uBandCount+uBandPhase));
  // glare inside the barrel and on the phone lens (optical, so before the sensor)
  vec3 lowR=pow(textureLod(uPanel,vec2(.5),9.).rgb,vec3(2.2));
  float low=lowR[ch];
  float ra=length(a0*vec2(.86,1.));
  float barrel=exp(-pow((ra-.90)/.30,2.))*low;
  float rubber=(.72+.28*vnoise(a0*140.))*smoothstep(1.34,.78,ra);
  float ambient=uRubber*rubber*(barrel+.0009);
  float veil=uVeil*pow(textureLod(uPanel,clamp(uvc,0.,1.),5.5)[ch],2.2)+uVeil*.35*low;
  vec2 gu=1.-clamp(uvc,0.,1.);
  float gh=pow(textureLod(uPanel,gu,3.5).g,5.)*uGhost*(ch==1?.14:.7);
  float E=(L+ambient+veil+gh)*uGain;
  // photon + read noise, Bayer domain
  vec2 g=gauss(gl_FragCoord.xy+uFrame*vec2(3.1,5.7));
  float sig=sqrt(max(E,0.)*.0032+.000022*uGain*uGain*.35+.000012);
  E+=g.x*sig*uNoise;
  o=vec4(max(E,0.),0.,0.,1.);
}
