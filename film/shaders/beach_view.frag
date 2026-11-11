#version 410
// The solver's state as the renderer wants it: surface height, depth, velocity (filterable, mipped),
// and how much foam and spray each cell holds.
//
// A bore in the solver is a step one cell wide, which no water is. Ahead of the step the sea is drawn up
// into it: a smooth concave face, shallow at the toe and steep under the lip. The water of that face has
// no foam on it yet, so a breaker comes out dark below and white above, as they are. Where the lip spills
// the foam runs some way down the face, further in some places than in others.
in vec2 uv; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
#include "common.glsl"
uniform sampler2D uS,uB,uM,uLaceN;
uniform vec4 uGrid;
uniform float uBlend;
const int FACE_N=24;
const float DRAW=.46;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  ivec2 top=textureSize(uS,0)-1;
  vec4 s=texelFetch(uS,p,0);
  vec4 b=texelFetch(uB,p,0);
  vec2 u=s.x>.002?s.yz/s.x:vec2(0.);
  float th=uGrid.x+(float(p.x)+.5)*uGrid.y;
  float r=exp(uGrid.z+(float(p.y)+.5)*uGrid.w);
  vec2 q=r*vec2(sin(th),cos(th));
  float cell=r*uGrid.y;
  float eta=s.x+s.w;
  float face=eta;
  float Fs=b.r;
  if(s.x>.03){
    int jb=0; float db=0.;
    for(int j=1;j<=FACE_N;j++){
      vec4 a=texelFetch(uS,ivec2(p.x,min(p.y+j,top.y)),0);
      if(a.x<.03||a.w>s.w+.35) break;     // sand or rock in the way: not a wave
      float dl=float(j)*cell;
      float e=a.x+a.w-DRAW*pow(dl,.6);
      if(e>face){ face=e; jb=j; db=dl; }
    }
    if(jb>0){
      float lb=log2(cell/(9./2048.));
      float n1=textureLod(uLaceN,q/43.+.3,max(lb-2.,0.)).r;
      float n2=textureLod(uLaceN,q/15.+.7,max(lb-1.,0.)).r;
      float spill=(.1+2.3*smoothstep(.3,.8,n1))*(.3+1.5*n2*n2)*smoothstep(.04,.5,face-eta);
      // what is on the lip, from behind the jump itself: the jump falls anywhere within a cell
      float Fc=0.;
      for(int k=1;k<=10;k+=3) Fc=max(Fc,texelFetch(uB,ivec2(p.x,min(p.y+jb+k,top.y)),0).r);
      Fs=max(Fs,Fc*(1.-smoothstep(0.,spill,db)));
    }
  }
  // close to, the water is thin and its fronts are the size of what is under them
  float fill=(face-eta)*smoothstep(13.,24.,r);
  face=eta+fill;
  // the lumps of the roller travel with the water
  vec4 m=texelFetch(uM,p,0);
  float lod=max(log2(cell/(17./2048.)),0.);
  float l1=textureLod(uLaceN,(q+m.xy)/17.,lod).r,l2=textureLod(uLaceN,(q+m.zw)/17.,lod).r;
  float lumps=mix(l1,l2,uBlend);
  float l3=mix(textureLod(uLaceN,(q+m.xy)/4.3+.2,lod+1.).r,textureLod(uLaceN,(q+m.zw)/4.3+.2,lod+1.).r,uBlend);
  float thick=min(b.r,1.)*smoothstep(.05,.45,s.x);
  // on a steep face a lump would only be extruded into a curtain: lumps ride the crest and the back
  float eE=texelFetch(uS,p+ivec2(1,0),0).x+texelFetch(uS,p+ivec2(1,0),0).w,eW=texelFetch(uS,p-ivec2(1,0),0).x+texelFetch(uS,p-ivec2(1,0),0).w;
  float eN=texelFetch(uS,p+ivec2(0,1),0).x+texelFetch(uS,p+ivec2(0,1),0).w,eQ=texelFetch(uS,p-ivec2(0,1),0).x+texelFetch(uS,p-ivec2(0,1),0).w;
  float sl=length(vec2(eE-eW,eN-eQ))/(2.*cell);
  float lift=thick*(.03+.42*lumps*lumps*lumps+.12*l3*l3)*smoothstep(10.,60.,r)/(1.+3.*sl*sl);
  o0=vec4(face+lift,s.x+fill,u);
  float spray=max(Fs-.45,0.)*smoothstep(.2,.7,s.x);
  // wetness is on or off per cell where the water last reached: take the corners off
  float W=0.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++)
    W+=texelFetch(uB,clamp(p+ivec2(i,j),ivec2(0),textureSize(uB,0)-1),0).g*float((2-abs(i))*(2-abs(j)));
  o1=vec4(Fs,W/16.,spray,fill);
}
